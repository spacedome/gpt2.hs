{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# OPTIONS_GHC -Wno-unused-binds #-}

import Data.List (maximumBy, sortBy)
import Data.Ord (Down (Down), comparing)
import Loader
import Model
import Numeric.LinearAlgebra (fromList, toList)
import Numeric.Natural
import System.Random (randomRIO)

topK :: V -> V
topK v = fromList (map f (toList v))
  where
    k = sortBy (comparing Down) (toList v) !! 200
    -- here 2 is the "tempurature"
    f x = if x > k then x / 2 else -1 / 0

-- simple max sampler for testing
sampleMax :: V -> IO Token
sampleMax x = return $ snd $ maximumBy (comparing fst) (zip (toList x) [0 ..])

sampleLogits :: V -> IO Int
sampleLogits v = do
  r <- randomRIO (0.0, 1.0)
  return $ findIndex r (scanl1 (+) (toList $ softmax $ topK v))
  where
    findIndex r cumProbs = length (takeWhile (<= r) cumProbs)

run :: GPT -> Natural -> [Token] -> IO [Token]
run model iter tokens = do
  next <- sampleLogits $ last $ forwardModel model tokens
  print next
  if iter == 0
    then return (tokens ++ [next])
    else run model (iter - 1) (tokens ++ [next])

main :: IO ()
main = do
  results <- readModel "model.safetensors"
  let model = case results of Right gpt -> gpt; Left err -> error err
  -- Tokens for "Hello, I am"
  generate <- run model 25 [15496, 11, 314, 716]
  print generate
