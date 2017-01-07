import unittest
import NAME.module1
import NAME.module2

class test_NAME(unittest.TestCase):

    def setUp(self):
        print('setting up!')

    def test_feature1(self):
        print('testing feature 1!')

    def test_feature2(self):
        print('testing feature 2!')

    def test_fail(self):
        self.assertEqual(3,1)

    def tearDown(self):
        print('we are done here!')

