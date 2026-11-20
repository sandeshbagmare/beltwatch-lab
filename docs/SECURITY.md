# Data and security boundaries

BeltWatch generates equipment footage locally. It has no camera input, file upload, account, remote inference service, or connection to equipment controls. After installation, the training command and saved replay can run without a network connection.

The command writes into a new directory chosen by the runner and refuses an existing path. It trains from generated frames and exports neural weights as safetensors. The replay contains its own clips and scores, with no external scripts or media requests. The public GitHub Pages site hosts that same experiment and its documentation.

Generated HTML is still a program your browser runs. If someone edits a saved replay, review it as you would any other changed program before opening it.

For a suspected issue, use the repository's GitHub reporting channel and describe the affected version and behavior. Keep credentials and private footage out of the report.
