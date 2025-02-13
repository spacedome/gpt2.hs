{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-unused-binds #-}

import Data.List (maximumBy, sortBy)
import Data.Ord (Down (Down), comparing)
import qualified Data.Text.IO as TIO
import Loader
import Model
import Numeric.LinearAlgebra (fromList, toList)
import Numeric.Natural
import System.Random (randomRIO)
import Token

topK :: V -> V
topK v = fromList (map f (toList v))
  where
    -- top 50 seems to work well here
    k = sortBy (comparing Down) (toList v) !! 50
    -- here 2 is the "tempurature"
    f x = if x > k then x / 2 else -1 / 0

-- simple max probability sampler for testing. Deterministic output
sampleMax :: V -> IO Token
sampleMax x = return $ snd $ maximumBy (comparing fst) (zip (toList x) [0 ..])

-- attempt to sample from the logits according to their softmax probabilities
-- this implementation might be a bit busted. sample only from top k
sampleLogits :: V -> IO Int
sampleLogits v = do
  r <- randomRIO (0.0, 1.0)
  return $ findIndex r (scanl1 (+) (toList $ softmax $ topK v))
  where
    findIndex r cumProbs = length (takeWhile (<= r) cumProbs)

run :: GPT -> TokenMap -> Natural -> [Token] -> IO [Token]
run model tm iter tokens = do
  next <- sampleLogits $ last $ forwardModel model tokens
  -- unicode output isn't handled correctly, not sure if its the print or the read
  TIO.putStrLn (token tm (tokens ++ [next]))
  if iter == 0
    then return (tokens ++ [next])
    else run model tm (iter - 1) (tokens ++ [next])

main :: IO ()
main = do
  putStrLn "λλμ: Now Loading..."
  tensors <- readModel "model.safetensors"
  vocab <- readVocab "vocab.json"
  let model = case tensors of Right gpt -> gpt; Left err -> error err
  let tokenMap = case vocab of Just tm -> tm; Nothing -> error "Couldn't parse vocab"
  -- Tokens for "Hello, I am"
  generate <- run model tokenMap 20 [15496, 11, 314, 716]
  TIO.putStrLn (token tokenMap generate)
  print generate
