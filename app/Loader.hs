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

-- simple sum type so we can load either vec or mat
data Tensor = T1 V | T2 M

-- generate a keymap based on the safetensor metadata
type TensorMap = KM.KeyMap Tensor

-- metadata for an individual tensor
data TensorMetadata = TensorMetadata
  { dtype :: String,
    shape :: [Int],
    dataOffsets :: (Int, Int)
  }
  deriving (Show, Generic, FromJSON, ToJSON)

-- entire safetensors file including unmapped raw tensor data
data SafeTensors = SafeTensors
  { metadata :: KM.KeyMap TensorMetadata,
    binaryData :: BL.ByteString
  }

-- we don't want to show the binary data, might as well have a pretty printer
instance Show SafeTensors where
  show safetensors = show $ encodePretty (metadata safetensors)

-- Parse tensor metadata from JSON segment of file
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
  -- the first 8 bytes are an uint specifiying length of JSON segment
  numBytes <- parseWord64 (BL.take 8 bs)
  -- the next N bytes can be decoded directly with aeson
  obj <- decode (BL.take (fromIntegral numBytes) (BL.drop 8 bs))
  -- this is the one key that isn't a tensor, easiest just to remove it
  let tensors = KM.delete (K.fromString "__metadata__") obj
  -- parse tensor metadata objects into our metadata type
  x <- mapM (parseMaybe parseTensorMetadata) tensors
  -- return metadata keymap along with remaining raw bytes containing tensor data
  return (SafeTensors x (BL.drop (8 + fromIntegral numBytes) bs))

-- parse a Word64 from the head of the file (encodes length of JSON segment)
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
  [n] -> T1 (n |> dataChunk)
  [n, m] -> T2 ((n >< m) dataChunk)
  [1, 1, n, m] -> T2 ((n >< m) dataChunk)
  _ -> undefined
  where
    (startpos, endpos) = bimap fromIntegral fromIntegral (dataOffsets meta)
    dataChunk = byteStringToFloats (BL.drop startpos (BL.take endpos bs))

getTensorMap :: SafeTensors -> TensorMap
getTensorMap ten = fmap (bytesToTensor (binaryData ten)) (metadata ten)

getMat :: TensorMap -> String -> Maybe M
getMat tm s = case KM.lookup (K.fromString s) tm of
  (Just (T2 m)) -> Just m
  _ -> Nothing

getVec :: TensorMap -> String -> Maybe V
getVec tm s = case KM.lookup (K.fromString s) tm of
  (Just (T1 v)) -> Just v
  _ -> Nothing

getTELayer :: TensorMap -> Maybe TokenEmbedding
getTELayer tm = do
  m <- getMat tm "wte.weight"
  return (TokenEmbedding (toRows m))

getPELayer :: TensorMap -> Maybe PositionEmbedding
getPELayer tm = do
  m <- getMat tm "wpe.weight"
  return (PositionEmbedding (toRows m))

getLayerNorm :: TensorMap -> String -> Maybe LayerNorm
getLayerNorm tm s = do
  w <- getVec tm (s ++ ".weight")
  b <- getVec tm (s ++ ".bias")
  return (LayerNorm w b)

getAttention :: TensorMap -> String -> Maybe Attention
getAttention tm layer = do
  aw <- getMat tm (layer ++ ".attn.c_attn.weight")
  ab <- getVec tm (layer ++ ".attn.c_attn.bias")
  pw <- getMat tm (layer ++ ".attn.c_proj.weight")
  pb <- getVec tm (layer ++ ".attn.c_proj.bias")
  return (Attention (tr aw) ab (tr pw) pb)

getMLP :: TensorMap -> String -> Maybe MLP
getMLP tm layer = do
  aw <- getMat tm (layer ++ ".mlp.c_fc.weight")
  ab <- getVec tm (layer ++ ".mlp.c_fc.bias")
  pw <- getMat tm (layer ++ ".mlp.c_proj.weight")
  pb <- getVec tm (layer ++ ".mlp.c_proj.bias")
  return (MLP (tr aw) ab (tr pw) pb)

getBlock :: TensorMap -> Int -> Maybe Block
getBlock tm i = do
  let prefix = "h." ++ show i
  le1 <- getLayerNorm tm (prefix ++ ".ln_1")
  le2 <- getLayerNorm tm (prefix ++ ".ln_2")
  at <- getAttention tm prefix
  mp <- getMLP tm prefix
  return (Block le1 at le2 mp)
