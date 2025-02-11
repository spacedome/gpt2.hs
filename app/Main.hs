{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Data.Aeson
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.Aeson.Key as K
import qualified Data.Aeson.KeyMap as KM
import Data.Aeson.Types (Parser, parseMaybe)
import Data.Bifunctor (bimap)
import Data.Binary.Get (Get, getFloatle, getWord64le, isEmpty, runGet)
import qualified Data.ByteString.Lazy as BL
import Data.List (transpose)
import Data.Word (Word64)
import Debug.Trace
import GHC.Generics
import Numeric.LinearAlgebra (Matrix, Vector, build, cmap, rows, scalar, size, sumElements, tr, (#>), (<>), (><), (|>))
import Numeric.LinearAlgebra.Data (fromColumns, fromRows, takesV, toRows, vjoin)
import Prelude hiding ((<>))

type V = Vector Float

type M = Matrix Float

-- Define a data structure to hold tensor metadata
data TensorMetadata = TensorMetadata
  { dtype :: String,
    shape :: [Int],
    dataOffsets :: (Int, Int)
  }
  deriving (Show, Generic, FromJSON, ToJSON)

-- Define a data structure to hold the parsed safetensors file
data SafeTensors = SafeTensors
  { metadata :: KM.KeyMap TensorMetadata,
    binaryData :: BL.ByteString
  }

instance Show SafeTensors where
  show safetensors = show $ encodePretty (metadata safetensors)

-- Parse tensor metadata from JSON
parseTensorMetadata :: Value -> Parser TensorMetadata
parseTensorMetadata = withObject "TensorMetadata" $ \obj -> do
  mdtype <- obj .: "dtype"
  mshape <- obj .: "shape"
  (i, j) <- obj .: "data_offsets"
  return
    ( TensorMetadata
        { shape = mshape,
          dataOffsets = (i, j),
          dtype = mdtype
        }
    )

readSafeTensors :: FilePath -> IO (Maybe SafeTensors)
readSafeTensors filePath = do
  contents <- BL.readFile filePath
  return (parseTensors contents)

parseTensors :: BL.ByteString -> Maybe SafeTensors
parseTensors bs = do
  numBytes <- parseWord64 (BL.take 8 bs)
  obj <- decode (BL.take (fromIntegral numBytes) (BL.drop 8 bs)) :: Maybe Object
  let tensors = KM.delete (K.fromString "__metadata__") obj
  x <- mapM (parseMaybe parseTensorMetadata) tensors
  return (SafeTensors x (BL.drop (8 + fromIntegral numBytes) bs))

-- Function to parse a Word64 from the head of a ByteString
parseWord64 :: BL.ByteString -> Maybe Word64
parseWord64 bs
  | BL.length bs >= 8 = Just $ runGet getWord64le bs
  | otherwise = Nothing

byteStringToFloats :: BL.ByteString -> [Float]
byteStringToFloats = runGet getFloats
  where
    getFloats :: Get [Float]
    getFloats = do
      empty <- isEmpty
      if empty
        then return []
        else do
          x <- getFloatle
          xs <- getFloats
          return (x : xs)

bytesToTensor :: BL.ByteString -> TensorMetadata -> Tensor
bytesToTensor bs meta = case shape meta of
  [] -> undefined
  [n] -> T1 (n |> dataChunk)
  [n, m] -> T2 ((n >< m) dataChunk)
  [1, 1, n, m] -> T2 ((n >< m) dataChunk)
  _ -> undefined
  where
    (startpos, endpos) = bimap fromIntegral fromIntegral (dataOffsets meta)
    dataChunk = byteStringToFloats (BL.drop startpos (BL.take endpos bs))

getTensorMap :: SafeTensors -> TensorMap
getTensorMap ten = fmap (bytesToTensor (binaryData ten)) (metadata ten)

getTELayer :: TensorMap -> TokenEmbeddingLayer
getTELayer tm = TokenEmbeddingLayer x
  where
    x = case KM.lookup (K.fromString "wte.weight") tm of
      Just (T2 m) -> toRows m
      _ -> undefined

getPELayer :: TensorMap -> PositionEmbeddingLayer
getPELayer tm = PositionEmbeddingLayer x
  where
    x = case KM.lookup (K.fromString "wpe.weight") tm of
      Just (T2 m) -> toRows m
      _ -> undefined

getLayerNorm :: TensorMap -> String -> LayerNorm
getLayerNorm tm s = LayerNorm w b
  where
    w = case KM.lookup (K.fromString (s ++ ".weight")) tm of
      Just (T1 v) -> v
      _ -> undefined
    b = case KM.lookup (K.fromString (s ++ ".bias")) tm of
      Just (T1 v) -> v
      _ -> undefined

getAttention :: TensorMap -> String -> Attention
getAttention tm layer = Attention aw ab pw pb
  where
    aw = case KM.lookup (K.fromString (layer ++ ".attn.c_attn.weight")) tm of
      Just (T2 v) -> tr v
      _ -> undefined
    ab = case KM.lookup (K.fromString (layer ++ ".attn.c_attn.bias")) tm of
      Just (T1 v) -> v
      _ -> undefined
    pw = case KM.lookup (K.fromString (layer ++ ".attn.c_proj.weight")) tm of
      Just (T2 v) -> tr v
      _ -> undefined
    pb = case KM.lookup (K.fromString (layer ++ ".attn.c_proj.bias")) tm of
      Just (T1 v) -> v
      _ -> undefined

getMLP :: TensorMap -> String -> MLPLayer
getMLP tm layer = MLP aw ab pw pb
  where
    aw = case KM.lookup (K.fromString (layer ++ ".mlp.c_fc.weight")) tm of
      Just (T2 v) -> tr v
      _ -> undefined
    ab = case KM.lookup (K.fromString (layer ++ ".mlp.c_fc.bias")) tm of
      Just (T1 v) -> v
      _ -> undefined
    pw = case KM.lookup (K.fromString (layer ++ ".mlp.c_proj.weight")) tm of
      Just (T2 v) -> tr v
      _ -> undefined
    pb = case KM.lookup (K.fromString (layer ++ ".mlp.c_proj.bias")) tm of
      Just (T1 v) -> v
      _ -> undefined

getBlock :: TensorMap -> Int -> BlockLayer
getBlock tm i = BlockLayer le1 at le2 mp
  where
    prefix = "h." ++ show i
    le1 = getLayerNorm tm (prefix ++ ".ln_1")
    le2 = getLayerNorm tm (prefix ++ ".ln_2")
    at = getAttention tm prefix
    mp = getMLP tm prefix

data Tensor = T1 V | T2 M deriving (Show)

type TensorMap = KM.KeyMap Tensor

newtype TokenEmbeddingLayer = TokenEmbeddingLayer [V]

newtype PositionEmbeddingLayer = PositionEmbeddingLayer [V]

data BlockLayer = BlockLayer LayerNorm Attention LayerNorm MLPLayer

data MLPLayer = MLP M V M V

data LayerNorm = LayerNorm V V

data Attention = Attention M V M V

data GPTModel = GPTModel
  { wpe :: PositionEmbeddingLayer,
    wte :: TokenEmbeddingLayer,
    blocks :: [BlockLayer]
  }

constructModel :: TensorMap -> GPTModel
constructModel tm =
  GPTModel
    { wpe = getPELayer tm,
      wte = getTELayer tm,
      blocks = reverse (getBlock tm <$> [0 .. 11])
    }

forwardLN :: LayerNorm -> V -> V
forwardLN (LayerNorm w b) x = y
  where
    n = fromIntegral (size x)
    mean = scalar (sumElements x / n)
    cent = x - mean
    varx = sumElements (cent * cent) / n
    fact = scalar (sqrt (varx + 1e-5))
    y = ((x - mean) / fact) * w + b

forwardAtToken :: Attention -> V -> ([V], [V], [V])
forwardAtToken (Attention w b _ _) x = (qh, kh, vh)
  where
    y = (w #> x) + b -- [3N]
    qkv = takesV [768, 768, 768] y
    (q, k, v) = (head qkv, (head . tail) qkv, (head . tail . tail) qkv)
    qh = takesV (replicate 12 64) q
    kh = takesV (replicate 12 64) k
    vh = takesV (replicate 12 64) v

forwardHead :: (M, M, M) -> M
forwardHead (q, k, v) = z
  where
    attnMatrix = tr q <> k * scalar (1 / 8) -- 1 / sqrt (size k)
    attnMasked = tril (rows attnMatrix) + attnMatrix
    attnSoftmax = fromRows (fmap softmax (toRows attnMasked))
    z = attnSoftmax <> tr v

forwardAttn :: Attention -> [V] -> [V]
forwardAttn at@(Attention _ _ w b) xs = z
  where
    (q, k, v) = unzip3 (fmap (forwardAtToken at) xs)
    qh = fmap fromColumns (transpose q)
    kh = fmap fromColumns (transpose k)
    vh = fmap fromColumns (transpose v)
    lm = fmap forwardHead (zip3 qh kh vh)
    y = fmap vjoin (transpose (fmap toRows lm))
    z = fmap ((+ b) . (w #>)) y

forwardMLP :: MLPLayer -> [V] -> [V]
forwardMLP (MLP wfc bfc wproj bproj) x = x3
  where
    x1 = fmap ((+ bfc) . (wfc #>)) x
    x2 = fmap gelu x1
    x3 = fmap ((+ bproj) . (wproj #>)) x2

forwardBlock :: BlockLayer -> [V] -> [V]
forwardBlock (BlockLayer l1 at l2 mp) xs = x5
  where
    x1 = fmap (forwardLN l1) xs
    x2 = zipWith (+) xs (forwardAttn at x1)
    x3 = fmap (forwardLN l2) x2
    x4 = zipWith (+) x2 (forwardMLP mp x3)
    x5 = (traceShow (fmap (takesV [5]) x4)) x4

softmax :: V -> V
softmax v = expv * scalar (1 / sumElements expv)
  where
    expv = cmap exp v

gelu :: V -> V
gelu x = 0.5 * x * (1 + tanh (sqrt (2 / pi) * (x + 0.044715 * x * x * x)))

-- Function to fill the upper triangular part of a matrix with -inf
tril :: Int -> M
tril n = build (n, n) (\i j -> if j > i then -1 / 0 else 0)

forward :: GPTModel -> Int -> [V]
forward model token = o
  where
    TokenEmbeddingLayer wtew = wte model
    PositionEmbeddingLayer wpew = wpe model
    emb1 = (wtew !! token) + head wpew
    emb2 = (wtew !! token) + (head . tail) wpew
    o = (foldr (.) id (forwardBlock <$> blocks model)) [emb1, emb2]

run :: Maybe SafeTensors -> Maybe String
run safeten = do
  t <- safeten
  let ten = getTensorMap t
  let model = constructModel ten
  let next = forward model 15496
  return (show (fmap (takesV [5]) next))

main :: IO ()
main = do
  results <- readSafeTensors "model.safetensors"
  -- case results of
  --   Just ten -> BL.putStr (encodePretty (metadata ten))
  --   Nothing -> putStrLn "error"
  -- putStrLn ""
  case run results of
    Just str -> putStrLn str
    Nothing -> putStrLn "error"
