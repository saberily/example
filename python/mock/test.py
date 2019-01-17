from unittest import TestCase
from unittest.mock import patch, Mock
from main import Calculator
import main
from mymodule import rm
from mymodule import RemovalService, UploadService
import os, os.path
import tempfile
import time


class TestCalculator(TestCase):
    def setUp(self):
        self.calc = Calculator()

    def test_sum(self):
        answer = self.calc.sum(2, 4)
        self.assertEqual(answer, 6)


class TestCalculator2(TestCase):
    @patch('main.Calculator.sum', return_value=9)
    def test_sum(self, sum):
        self.assertEqual(sum(2,3), 9)


def mock_sum(a, b):
    return a + b


class TestCalculator3(TestCase):
    @patch('main.Calculator.sum', side_effect=mock_sum)
    def test_sum(self, sum):
        self.assertEqual(sum(2,3), 5)
        self.assertEqual(sum(7,3), 10)


class TestBlog(TestCase):
    @patch('main.Blog')
    def test_blog_posts(self, MockBlog):
        blog = MockBlog()

        blog.posts.return_value = [
            {
                'userId': 1,
                'id': 1,
                'title': 'Test Title',
                'body': 'nothing',
            },
        ]

        response = blog.posts()
        self.assertIsNotNone(response)
        self.assertIsInstance(response[0], dict)

        # Additional assertions
        assert MockBlog is main.Blog # The mock is equivalent to the original
        assert MockBlog.called # The mock wasP called
        blog.posts.assert_called_with() # We called the posts method with no arguments
        blog.posts.assert_called_once_with() # We called the posts method once with no arguments
        # blog.posts.assert_called_with(1, 2, 3) - This assertion is False and will fail since we called blog.posts with no arguments

        #blog.reset_mock() # Reset the mock object
        #blog.posts.assert_not_called() # After resetting, posts has not been called.


class RmTestCase(TestCase):
    print(tempfile.gettempdir())
    tmpfilepath = os.path.join(tempfile.gettempdir(), "tmp-testfile")

    def setUp(self):
        with open(self.tmpfilepath, "w") as f:
            f.write("Delete me!")

    def test_rm(self):
        #remove the file
        rm(self.tmpfilepath)
        #test that it was actually removed
        self.assertFalse(os.path.isfile(self.tmpfilepath), "Failed to remove the file.")


class RmTestCase2(TestCase):
    @patch('mymodule.os')
    def test_rm(self, mock_os):
        rm("any path")
        #test that rm called os.remove with the right parameters
        mock_os.remove.assert_called_with("any path")


class RmTestCase3(TestCase):

    @patch('mymodule.os.path')
    @patch('mymodule.os')
    def test_rm(self, mock_os, mock_path):
        mock_path.isfile.return_value = False
        
        rm("any path")

        self.assertFalse(mock_os.remove.called, "Failed to not remove the file.")

        mock_path.isfile.return_value = True

        rm("any path")

        mock_os.remove.assert_called_with("any path")


class RemovalServiceTestCase(TestCase):

    #patch修饰之后mymodule.os.path已经变成假的函数即mock_path
    #patch修饰之后mymodule.os已经变成假的函数即mock_os
    #之后RemovalService中的os.path和os调用都为假调用，需要手动赋值
    @patch('mymodule.os.path')
    @patch('mymodule.os')
    def test_rm(self, mock_os, mock_path):
        reference = RemovalService()

        mock_path.isfile.return_value = False
        
        reference.rm("any path")

        self.assertFalse(mock_os.remove.called, "Failed to not remove the file.")

        mock_path.isfile.return_value = True

        reference.rm("any path")

        mock_os.remove.assert_called_with("any path")


class UploadServiceTestCase(TestCase):

    @patch.object(RemovalService, 'rm')
    def test_upload_complete(self, mock_rm):
        removal_service = RemovalService()
        reference = UploadService(removal_service)

        reference.upload_complete("my uploaded file")

        mock_rm.assert_called_with("my uploaded file")

        removal_service.rm.assert_called_with("my uploaded file")


        