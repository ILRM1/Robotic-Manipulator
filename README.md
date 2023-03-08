# Catching robot

<p align="center"><img width="80%" src="https://user-images.githubusercontent.com/53123394/223743206-21e14333-660c-4276-9ccd-8b2215b6c9de.png"/>

The roll_simulation.ttt is CoppeliaSim simulator file. Please run python files as follows:

1. generate_noise.py (for applying Perlin noises in simulator)
2. generate_data.py (creating images and trajectories)
3. preprocess.py (preprocess trajectories for batch and resize images)
4. train_detection.py (training for Detection network)
5. train_transformer.py (training for Embedding network and transformer)
