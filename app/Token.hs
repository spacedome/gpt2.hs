module Token where

import Data.Aeson
import Data.Aeson.Key (toText)
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.ByteString.Lazy as BL
import qualified Data.Map as Map
import qualified Data.Text as T
import Model
import Data.Maybe (mapMaybe)

type TokenMap = Map.Map Token T.Text

-- This sorta works but has weird unicode issues around whitespace
-- Not something I want to figure out, so decode it with python tiktoken

parseStringIntJSON :: BL.ByteString -> Maybe TokenMap
parseStringIntJSON json = do
  keyMap <- decode json
  let pairings = map (\(k, v) -> (v, toText k)) (KeyMap.toList keyMap)
  return $ Map.fromList pairings

readVocab :: String -> IO (Maybe TokenMap)
readVocab filePath = do
  contents <- BL.readFile filePath
  return (parseStringIntJSON contents)

token :: TokenMap -> [Token] -> String
token tm t = (T.unpack . removeSpecial . T.concat) (mapMaybe (`Map.lookup` tm) t)


removeSpecial :: T.Text -> T.Text
removeSpecial = T.replace (T.singleton '\x0120') (T.singleton ' ')
