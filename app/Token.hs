module Token where

import Data.Aeson
import Data.Aeson.Key (toText)
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.ByteString.Lazy as BL
import qualified Data.Map as Map
import qualified Data.Text as T
import Model

type TokenMap = Map.Map Token String

parseStringIntJSON :: BL.ByteString -> Maybe TokenMap
parseStringIntJSON json = do
  keyMap <- decode json
  let pairings = map (\(k, v) -> (v, T.unpack (toText k))) (KeyMap.toList keyMap)
  return (Map.fromList pairings)

readVocab :: String -> IO (Maybe TokenMap)
readVocab filePath = do
  contents <- BL.readFile filePath
  return (parseStringIntJSON contents)
