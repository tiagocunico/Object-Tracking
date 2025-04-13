import numpy as np
import cv2
import os
import heapq

def inRange(cordinates, limits):
    x, y = cordinates
    X_Limit, Y_Limit = limits
    return 0 <= x and x < X_Limit and 0 <= y and y < Y_Limit

def calculate_frame_difference(prev_frame, curr_frame):
    """Calcula a diferença total entre dois frames"""
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.max(diff)

def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):
    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)

    old_frame = old_frame / 255
    new_frame = new_frame / 255

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)
    fy = cv2.filter2D(old_frame, -1, kernel_y)
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)

    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    if feature_list is not None:
        for feature in feature_list:
            j, i = feature.ravel()
            i, j = int(i), int(j)

            if i-w >= 0 and i+w+1 <= old_frame.shape[0] and j-w >= 0 and j+w+1 <= old_frame.shape[1]:
                I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

                if len(I_x) > 0 and len(I_y) > 0 and len(I_t) > 0:
                    b = np.reshape(I_t, (I_t.shape[0],1))
                    A = np.vstack((I_x, I_y)).T

                    if A.shape[0] >= 2:
                        try:
                            U = np.matmul(np.linalg.pinv(A), b)
                            u[i,j] = U[0][0]
                            v[i,j] = U[1][0]
                        except:
                            pass

    return (u, v)

def process_video(input_path, output_path, window_size=3, min_quality=0.01):
    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    diff_frames_dir = "./Results/MaiorDiferencaSoro"
    os.makedirs(diff_frames_dir, exist_ok=True)
    
    # Data structure to keep top 10 different frames
    top_diffs = []
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Reduce speed by half
    new_fps = original_fps / 2
    
    # Video writer for optical flow
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_number = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds
        
        # Calculate frame difference
        diff = calculate_frame_difference(prev_gray, curr_gray)
        
        # Calculate optical flow
        u, v = optical_flow(prev_gray, curr_gray, window_size, min_quality)
        
        # Draw optical flow on frame with 200% larger arrows
        flow_frame = curr_frame.copy()
        for i in range(0, curr_gray.shape[0], 5):
            for j in range(0, curr_gray.shape[1], 5):
                if abs(u[i,j]) > 0.1 or abs(v[i,j]) > 0.1:
                    start_point = (j, i)
                    end_point = (int(j + 10 * u[i,j]), int(i + 10 * v[i,j]))
                    if inRange(end_point, (width, height)):
                        cv2.arrowedLine(flow_frame, start_point, end_point, 
                                       (0, 255, 0), 2, tipLength=0.3)
        
        # Write each frame twice to halve the speed
        out.write(flow_frame)
        out.write(flow_frame)
        
        # Store frame difference information
        if len(top_diffs) < 10:
            heapq.heappush(top_diffs, (diff, frame_number, timestamp, flow_frame))
        else:
            if diff > top_diffs[0][0]:
                heapq.heappop(top_diffs)
                heapq.heappush(top_diffs, (diff, frame_number, timestamp, flow_frame))
        
        # Update previous frame
        prev_gray = curr_gray.copy()
        frame_number += 1
    
    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save top 10 different frames and create metadata file
    save_top_differences(top_diffs, diff_frames_dir)

def save_top_differences(top_diffs, output_dir):
    # Sort by difference in descending order
    top_diffs_sorted = sorted(top_diffs, key=lambda x: x[0], reverse=True)
    
    # Prepare metadata content
    metadata = []
    
    for idx, (diff, frame_num, timestamp, frame) in enumerate(top_diffs_sorted):
        # Save frame as image
        frame_path = os.path.join(output_dir, f"frame_{frame_num:04d}_diff_{diff:.0f}.png")
        cv2.imwrite(frame_path, frame)
        
        # Add to metadata
        metadata.append(f"Rank {idx+1}: Frame {frame_num} | Timestamp: {timestamp:.2f}s | Diferença: {diff:.0f}")
    
    # Write metadata file
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("Top 10 frames com maior diferença:\n")
        f.write("---------------------------------\n")
        f.write("\n".join(metadata))
    
    print(f"Frames salvos em: {output_dir}")
    print(f"Metadados salvos em: {metadata_path}")

# Process the video
input_video = "./Inputs/soro.mp4"
output_video = "./Results/fluxosoro.mp4"
process_video(input_video, output_video)