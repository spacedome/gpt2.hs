module Token where

import Data.Aeson
import Data.Aeson.KeyMap (KeyMap)
import qualified Data.Aeson.KeyMap as KeyMap
import Data.Aeson.Key (toText)
import Data.ByteString.Lazy (ByteString)
import Data.ByteString.Lazy.Char8 (pack)
import qualified Data.Text as T

swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

type StringIntMap = KeyMap Int

-- Function to parse JSON and convert to list of tuples
parseStringIntJSON :: ByteString -> Maybe [(Int, String)]
parseStringIntJSON json = do
    keyMap <- decode json
    return $ map (\(k, v) -> (v, T.unpack (toText k))) (KeyMap.toList keyMap)


imain :: IO ()
imain = do
    let json = "{\"foo\": 42, \"bar\": 99, \"baz\": 123}"
    case parseStringIntJSON (pack json) of
        Just pairs -> print pairs
        Nothing   -> putStrLn "Failed to parse JSON."
