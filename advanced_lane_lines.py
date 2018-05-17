import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

dist_pickle = pickle.load(open( "calibrated_data.p", "rb" ))

def cal_undistort(img, mtx=dist_pickle['mtx'], dist=dist_pickle['dist']):
    return cv2.undistort(img, mtx, dist, None, mtx)

def pipeline(image, sx_thresh=(20, 100), s_thresh=(170, 255), l_thresh=(30, 255), sobel_kernel=3):

    # first we take a copy of the source iamge
    image_copy = np.copy(image)

    # convert RGB image to HLS color space.
    # HLS more reliable when it comes to find out lane lines
    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Next, we apply Sobel operator in X direction and calculate scaled derivatives.
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Next, we generate a binary image based on sx_thresh values.
    thresh_min = sx_thresh[0]
    thresh_max = sx_thresh[1]
    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Next, we generated a binary image using S component of our HLS color scheme and
    # provided S threshold
    s_binary = np.zeros_like(s_channel)
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Next, we generated a binary image using S component of our HLS color scheme and
    # provided S threshold
    l_binary = np.zeros_like(l_channel)
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # finally, return the combined binary image
    binary = np.zeros_like(sobel_x_binary)
    binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

    return binary

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):       

        self.left_fit = None
        self.right_fit = None

        self.MAX_BUFFER_SIZE = 12

        self.buffer_index = 0
        self.iter_counter = 0

        self.buffer_left = np.zeros((self.MAX_BUFFER_SIZE, 720))
        self.buffer_right = np.zeros((self.MAX_BUFFER_SIZE, 720))

        self.src, self.dst = self._build_perspective_transformer()

    @staticmethod
    def _build_perspective_transformer():
        """
        :return: src, dst
        """
        corners = np.float32([[252, 697], [585, 456], [700, 456], [1061, 690]])
        new_top_left = np.array([corners[0, 0], 0])
        new_top_right = np.array([corners[3, 0], 0])
        offset = [40, 0]

        src = np.float32([corners[0], corners[1], corners[2], corners[3]])
        dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

        return src, dst

    def lane_extractor(self, binary_warped):
        """
        :param binary_warped:
        :return: fit_leftx, fit_rightx
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :, 0], axis=0)

        # get midpoint of the histogram
        midpoint = np.int(histogram.shape[0] // 2)

        # get left and right halves of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # based on number of events, we calculate hight of a window
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Extracts x and y coordinates of non-zero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set current x coordinated for left and right
        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 75
        min_num_pixels = 35

        # save pixel ids in these two lists
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > min_num_pixels:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > min_num_pixels:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        fit_leftx = self.left_fit[0] * fity ** 2 + self.left_fit[1] * fity + self.left_fit[2]
        fit_rightx = self.right_fit[0] * fity ** 2 + self.right_fit[1] * fity + self.right_fit[2]

        return fit_leftx, fit_rightx
        
    def calculate_road_info(self, image_size, left_x, right_x):
        """
        This method calculates left and right road curvature and off of the vehicle from the center
        of the lane
        :return: Left and right curvatures of the lane and off of the vehicle from the center of the lane
        """
        # first we calculate the intercept points at the bottom of our image
        left_intercept = self.left_fit[0] * image_size[0] ** 2 + self.left_fit[1] * image_size[0] + self.left_fit[2]
        right_intercept = self.right_fit[0] * image_size[0] ** 2 + self.right_fit[1] * image_size[0] + self.right_fit[2]

        # Next take the difference in pixels between left and right interceptor points
        road_width_in_pixels = right_intercept - left_intercept
        assert road_width_in_pixels > 0, 'Road width in pixel can not be negative'

        # Since average highway lane line width in US is about 3.7m
        # we calculate length per pixel in meters
        meters_per_pixel_x_dir = 3.7 / road_width_in_pixels
        meters_per_pixel_y_dir = 30 / road_width_in_pixels

        # Recalculate road curvature in X-Y space
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, left_x * meters_per_pixel_x_dir, 2)
        right_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, right_x * meters_per_pixel_x_dir, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * meters_per_pixel_y_dir + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])

        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * meters_per_pixel_y_dir + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])

        # Next, we can lane deviation
        calculated_center = (left_intercept + right_intercept) / 2.0
        lane_deviation = (calculated_center - image_size[1] / 2.0) * meters_per_pixel_x_dir

        return left_curverad, right_curverad, lane_deviation

    @staticmethod
    def fill_lane_lines(image, fit_left_x, fit_right_x):
        """
        This utility method highlights correct lane section on the road
        :return: The input image with highlighted lane line.
        """
        copy_image = np.zeros_like(image)
        fit_y = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])

        pts_left = np.array([np.transpose(np.vstack([fit_left_x, fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, fit_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))

        return copy_image

    def merge_images(self, binary_img, src_image):
        """
        This utility method merges merges two images
        :return: The Original image with highlighted lane segment.
        """
        copy_binary = np.copy(binary_img)
        copy_src_img = np.copy(src_image)

        copy_binary_pers = warper(copy_binary, self.dst, self.src)
        result = cv2.addWeighted(copy_src_img, 1, copy_binary_pers, 0.3, 0)

        return result

    def process(self, image):
        """
        This method takes an image as an input and produces an image with
        1. Highlighted lane line
        2. Left and right lane curvatures
        3. Vehicle offset of the center of the lane
        :return: Annotated image with lane line details
        """
        image = np.copy(image)
        undistorted_image = cal_undistort(image)
        warped_image = warper(undistorted_image, self.src, self.dst)
        binary_image = pipeline(warped_image)

        fit_leftx, fit_rightx = self.lane_extractor(binary_image)

        self.buffer_left[self.buffer_index] = fit_leftx
        self.buffer_right[self.buffer_index] = fit_rightx

        self.buffer_index += 1
        self.buffer_index %= self.MAX_BUFFER_SIZE

        if self.iter_counter < self.MAX_BUFFER_SIZE:
            self.iter_counter += 1
            ave_left = np.sum(self.buffer_left, axis=0) / self.iter_counter
            ave_right = np.sum(self.buffer_right, axis=0) / self.iter_counter
        else:
            ave_left = np.average(self.buffer_left, axis=0)
            ave_right = np.average(self.buffer_right, axis=0)

        left_curvature, right_curvature, calculated_deviation = self.calculate_road_info(image.shape, ave_left,
                                                                                         ave_right)
        curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curvature, right_curvature)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image, curvature_text, (100, 50), font, 1, (220, 220, 220), 2)

        deviation_info = 'Lane Deviation: {:.3f} m'.format(calculated_deviation)
        cv2.putText(image, deviation_info, (100, 100), font, 1, (220, 220, 220), 2)

        filled_image = self.fill_lane_lines(binary_image, ave_left, ave_right)

        merged_image = self.merge_images(filled_image, image)

        return merged_image
    
    