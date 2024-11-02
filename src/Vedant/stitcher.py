import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        # Load images from the specified path
        image_paths = sorted(glob.glob(os.path.join(path, '*')))
        print(f'Found {len(image_paths)} images for stitching.')

        # Read and filter valid images
        images = [cv2.imread(img_path) for img_path in image_paths if cv2.imread(img_path) is not None]
        
        if len(images) < 2:
            print("Error: At least two images are required for stitching.")
            return None, []

        # First image to start
        panaroma = images[0]
        homographies = []

        # Iterate through images and compute homographies
        for i in range(1, len(images)):
            current_image = images[i]

            # Detect and match features between the current panaroma and the next image
            kp1, d1 = self.extract_features(panaroma)
            kp2, d2 = self.extract_features(current_image)
            matches = self.match_keypoints(d1, d2)

            # Skip if there are not enough matches
            if len(matches) < 4:
                print(f"Warning: Not enough matches between image {i-1} and image {i}. Skipping.")
                continue

            # Get matched points and compute the homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            H = self.calculate_homography(src_pts, dst_pts)
            homographies.append(H)

            # Warp the current image and stitch it to the panaroma
            panaroma = self.warp_and_combine(panaroma, current_image, H)

        return panaroma, homographies

    def extract_features(self, image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_keypoints(self, d1, d2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(d1, d2)
        return sorted(matches, key=lambda x: x.distance)

    def calculate_homography(self, src_pts, dst_pts):
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            x_prime, y_prime = dst_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)

        # Perform Singular Value Decomposition (SVD) to compute homography
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]  # Normalize to make the last element 1

    def warp_and_combine(self, img1, img2, H):
        # Get dimensions of both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Define the corners of img1 and img2
        corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

        # Transform corners of img2 using the homography H
        transformed_corners_img2 = cv2.perspectiveTransform(corners_img2, H)

        # Combine corners from img1 and transformed corners from img2 to determine the bounding box
        all_corners = np.concatenate((corners_img1, transformed_corners_img2), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Calculate translation to keep the panorama in positive coordinate space
        translation = [-xmin, -ymin]
        translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

        # Warp img2 using the combined translation and homography
        panorama_size = (xmax - xmin, ymax - ymin)
        result_img = cv2.warpPerspective(img2, translation_matrix @ H, panorama_size)

        # Paste img1 onto the result image at the correct location
        result_img[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]] = img1

        return result_img

