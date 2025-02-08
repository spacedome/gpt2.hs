{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

import qualified Data.ByteString.Lazy as BL
import Data.Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Key as K
import Data.Aeson.Types (parseMaybe, Parser)
import Data.Binary.Get (Get, getWord64le, getFloatle, runGet, runGetOrFail, isEmpty)
import Data.Word (Word64)
import GHC.Generics
import Data.Aeson.Encode.Pretty (encodePretty)

-- Define a data structure to hold tensor metadata
data TensorMetadata = TensorMetadata
  { dtype :: String
  , shape :: [Int]
  , dataOffsets :: (Int, Int)
  } deriving (Show, Generic, FromJSON, ToJSON)

-- Define a data structure to hold the parsed safetensors file
data SafeTensors = SafeTensors
  { metadata :: KM.KeyMap TensorMetadata
  , binaryData :: BL.ByteString
  } deriving (Show)

-- Parse tensor metadata from JSON
parseTensorMetadata :: Value -> Parser TensorMetadata
parseTensorMetadata = withObject "TensorMetadata" $ \obj -> do
  mdtype <- obj .: "dtype"
  mshape <- obj .: "shape"
  (i, j) <- obj .: "data_offsets"
  return (TensorMetadata {shape = mshape
  , dataOffsets = (i, j)
  , dtype = mdtype
  })

readSafeTensors :: FilePath -> IO (Maybe SafeTensors)
readSafeTensors filePath = do
  contents <- BL.readFile filePath;
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
  | otherwise        = Nothing

byteStringToFloats :: BL.ByteString -> Either String [Float]
byteStringToFloats bs = case runGetOrFail getFloats bs of
    Left (_, _, err) -> Left err
    Right (_, _, xs) -> Right xs
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


-- Example usage
main :: IO ()
main = do
  results <- readSafeTensors "test.tensor"
  case results of
    Just ten -> BL.putStr (encodePretty (metadata ten))
    Nothing -> putStrLn "error"
