{-# LANGUAGE OverloadedStrings #-}

import Model
import Loader
import Data.List ( maximumBy)
import Data.Ord (comparing)
import Numeric.LinearAlgebra (toList)

run :: Either String GPT -> Either String String
run model = do
  gpt <- model
  let next = forwardModel gpt [15496, 11] -- [616]
  let x = snd $ maximumBy (comparing fst) (zip (toList (last next)) [0..])
  return (show x)
  -- return (show (fmap (takesV [5]) next))

main :: IO ()
main = do
  results <- readModel "model.safetensors"
  -- case results of
  --   Just ten -> BL.putStr (encodePretty (metadata ten))
  --   Nothing -> putStrLn "error"
  -- putStrLn ""
  case run results of
    Right str -> putStrLn str
    Left str -> putStrLn ("error : " ++ str  )
