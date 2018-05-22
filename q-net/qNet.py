"""
Neural network using Keras (called by q_net_keras)
.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, merge, Activation, Conv2D, MaxPooling2D, Reshape, Permute

class NN():
    """
    Deep Q-learning network using Keras

    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    action_as_input : Boolean
        Whether the action is given as input or as output
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state,
                 action_as_input=False):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        inputs=[]

        dim = self._input_dimensions[0]
        input = Input(shape=dim)

        inputs.append(input)

        x = Flatten()(input)
        # we stack a deep fully-connected network on top
        x = Dense(7, activation='relu')(x)
        x = Dense(7, activation='relu')(x)

        out = Dense(self._n_actions, activation="linear")(x)

        model = Model(input=inputs, output=out)
        layers=model.layers

        # Grab all the parameters together.
        params = [ param
                    for layer in layers
                    for param in layer.trainable_weights ]

        return model, params

if __name__ == '__main__':
    pass

