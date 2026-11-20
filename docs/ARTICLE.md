# The conveyor model, and the baseline that beat it

Most of the time, the next frame of a conveyor video is easy to guess. The parcels move a little farther along the same path. A jam changes that pattern. So does a sudden reversal. The challenge is to learn enough about normal movement that these changes stand out, without training on examples of every possible fault.

BeltWatch Lab makes that problem small enough to inspect on a laptop. It generates its own conveyor footage, trains a 15,136-parameter model on normal clips, and produces a browser replay of what the model found. The footage is deliberately simple. That makes errors easier to trace, but it also gives a well-chosen baseline a considerable advantage.

## Predicting a representation

A small convolutional encoder turns each image into a vector of 16 values. A predictor looks at the previous four vectors and estimates the next one. When the next image arrives, the encoder turns that image into another vector. Their squared difference becomes the anomaly score.

BeltWatch borrows its training objective from LeWorldModel. The paper learns an encoder and a predictor together with a loss that rewards correct next-embedding predictions. Its larger model uses action inputs for planning. BeltWatch runs a smaller experiment on video alone and measures fault detection. [LeWorldModel](https://arxiv.org/abs/2603.19312)

There is a catch in learning both sides of a prediction. If the encoder maps every image to the same vector, the predictor can be accurate without understanding the belt at all. A low training loss then looks reassuring for the wrong reason.

## The first run took that shortcut

The initial full development run nearly collapsed. Different frames produced almost identical representations, and faults barely changed the score. The model passed the software tests, but the benchmark showed that it was not doing the intended job. A test can establish causality or correct array shapes without proving that training learned useful features.

The final encoder uses group normalization within each frame, and the predictor learns a correction to the most recent representation. Its loss also includes a Gaussian regularizer called SIGReg. This encourages the encoded vectors to retain a useful distribution of values instead of becoming constant. A regression test now checks that training on normal conveyor footage retains measurable variation.

The benchmark includes the same model trained without SIGReg, starting from the same weights and using the same minibatch order. Its representations still almost collapse. Their average channel spread is about 0.002, compared with 0.947 for the regularized model. Removing the regularizer also lowers fault recall from 83.8% to 11.7% in the reference run.

## Keeping the comparison honest

Training uses 160 normal scenes. Threshold calibration uses another 72 normal scenes. The final test set has 80 new scenes. Each is rendered as a normal clip plus a jammed version, a reversed version, and one with debris. The variants share exactly the same frames before the event starts. A detector cannot get credit simply for reacting to an unrelated difference in the background.

The development pilot informed the model choices. The final calibration and test scenes were drawn with fresh seeds after those choices were fixed. Each detector receives its own threshold from normal calibration clips, and only an alert during the actual fault counts as a detection.

The learned model detected 201 of the 240 fault clips and alerted on 7 of the 80 normal clips. It found every reversal and 77 of 80 jams. It found 44 of 80 debris events. The replay includes a debris example that stays below the threshold. Keeping that miss visible is more informative than showing only the strongest detection.

## The simplest motion rule won

The strongest comparison estimates horizontal motion from the first four frames. It tries shifts from three pixels left to three pixels right, picks the best fit, and then expects that shift to continue. A stopped belt, reversed belt, or new patch of debris creates an error against that expectation.

This detector found all 240 faults, while alerting on 6 normal clips. It also ranked every fault above every normal clip. The simulator moves the belt by the same horizontal distance each frame. That is exactly the motion the baseline tries to fit.

Regularization helped this model on the current simulator. Testing a changing camera, variable belt speed, or gradual equipment wear would show whether that result extends to other settings.

## What to inspect next

Choose a fault in the live demo and move through the event frame by frame. The browser plays scores saved by the Python run. Compare each detector with its own threshold because the errors use different scales. The saved metrics cover the complete test set. [Live demo](https://sandeshbagmare.github.io/beltwatch-lab/) · [Saved metrics](assets/metrics.json)
