class SoftMinLayer:
    def __init__(self):
        """

        :param input_size:
        :param output_size:
        """
        self.input_size = input_size
        self.output_size = output_size
        # layer weights
        self.W = None
        # layer biases
        self.W_0 = None
        self.set_weights(np.random.normal(loc=0, scale=1, size=(output_size, input_size)),
                         np.random.normal(loc=0, scale=1, size=(output_size, 1)))
        # layer input holder
        self.current_input = None
        # layer output holder
        self.current_output = None
        # derivative of Loss w.r.t. inputs
        self.dL_dinput = None
        # derivative of Loss w.r.t. weights
        self.dL_dW = None
        # derivative of Loss w.r.t. biases
        self.dL_dW_0 = None