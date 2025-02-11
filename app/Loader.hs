{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Loader where

import Data.Aeson
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.Aeson.Key as K
import qualified Data.Aeson.KeyMap as KM
import Data.Aeson.Types (Parser, parseMaybe)
import Data.Bifunctor (bimap)
import Data.Binary.Get (Get, getFloatle, getWord64le, isEmpty, runGet)
import qualified Data.ByteString.Lazy as BL
import Data.Word (Word64)
import GHC.Generics
import Model
import Numeric.LinearAlgebra (tr, (><), (|>))
import Numeric.LinearAlgebra.Data (toRows)
import Prelude hiding ((<>))

data Tensor = T1 V | T2 M deriving (Show)

type TensorMap = KM.KeyMap Tensor

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

getTELayer :: TensorMap -> TokenEmbedding
getTELayer tm = TokenEmbedding x
  where
    x = case KM.lookup (K.fromString "wte.weight") tm of
      Just (T2 m) -> toRows m
      _ -> undefined

getPELayer :: TensorMap -> PositionEmbedding
getPELayer tm = PositionEmbedding x
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

getMLP :: TensorMap -> String -> MLP
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

getBlock :: TensorMap -> Int -> Block
getBlock tm i = Block le1 at le2 mp
  where
    prefix = "h." ++ show i
    le1 = getLayerNorm tm (prefix ++ ".ln_1")
    le2 = getLayerNorm tm (prefix ++ ".ln_2")
    at = getAttention tm prefix
    mp = getMLP tm prefix
