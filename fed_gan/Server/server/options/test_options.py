from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes val options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--name', type=str, default='gan_city',help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of val examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result trainB')
        parser.add_argument('--phase', type=str, default='val', help='images, val, val, etc')
        # Dropout and Batchnorm has different behavioir during training and val.
        parser.add_argument('--eval', action='store_true', help='use eval mode during val time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many val trainB to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.add_argument('--client_num', type=int, default=1)
        parser.add_argument('--port', type=int)

        self.isTrain = False
        return parser
