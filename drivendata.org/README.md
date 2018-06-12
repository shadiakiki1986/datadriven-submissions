# datadriven-submissions
My submissions playing with https://www.drivendata.org/

Check individual folders for details.

Here is the chronological order of competitions tackled

- 46-box-plots for education
- 2-blood donations
- 1-UN millenium development goals
- 44-disease-prediction


Summary of results

- 46-box-plots for education
  - applied sklearn models + keras neural network
  - achieved rank 10 out of 588 competitors with score 0.69

- 2-blood donations
  - only sklearn models since not enough data
  - turns out that "ground truth" was public and found
  - ranked 1236 out of 4399 competitors with a score of 0.58
  - didn't really try harder for better results since most leaderboard submissions were close to each other anyway

- 1-UN millenium development goals
  - tried random forest but couldn't beat simple autoregression
  - ranked 13 out of 1738 with score 0.0511

- 44-disease-prediction
  - built keras architecture similar to [faceswap](https://github.com/deepfakes/faceswap) and [neural-style](https://github.com/jcjohnson/neural-style)
  - LSTM-based Autoencoder on features and on target, with 2-layer MLP in the middle
  - trained complete network of above altogether, optimizing at the same time:
    - autoencoder on features
    - autoencoder on target
    - MSE of error between encoded features through MLP and encoded target
  - achieved rank 35 out of 3500 competitors with score 18.75
