
from streamlit_image_coordinates import streamlit_image_coordinates
import tempfile
import numpy as np
import streamlit as st
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect, transform_points
from scipy.spatial import distance
import networkx as nx

def is_obstructed(start_pos, end_pos, opponent_positions, threshold=20):
    """Vérifie si le chemin entre start_pos et end_pos est obstrué par un joueur de l'équipe adverse."""
    for opp_pos in opponent_positions:
        dist = np.abs(np.cross(np.array(end_pos) - np.array(start_pos), np.array(start_pos) - np.array(opp_pos))) / np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        if dist < threshold:
            return True
    return False

def plot_shortest_path_to_goal(player_positions, player_team_indices, ball_holder_index, goal_position, frame, opponent_positions):
    # Créer un graphe pondéré
    G = nx.Graph()
    
    # Ajouter les noeuds (joueurs) du graphe
    for i, pos in enumerate(player_positions):
        if player_team_indices[i] == player_team_indices[ball_holder_index]:  # Seuls les joueurs de l'équipe du détenteur du ballon
            G.add_node(i, pos=pos)
    
    # Ajouter les arêtes (distances) entre les joueurs de la même équipe, en évitant les obstacles
    for i in G.nodes:
        for j in G.nodes:
            if i != j and not is_obstructed(player_positions[i], player_positions[j], opponent_positions):
                dist = distance.euclidean(player_positions[i], player_positions[j])
                G.add_edge(i, j, weight=dist)
    
    # Ajouter un nœud pour le but
    G.add_node('goal', pos=goal_position)
    
    # Connecter chaque joueur de l'équipe au but, sans obstacle
    for i in G.nodes:
        if i != 'goal' and not is_obstructed(player_positions[i], goal_position, opponent_positions):
            dist_to_goal = distance.euclidean(player_positions[i], goal_position)
            G.add_edge(i, 'goal', weight=dist_to_goal)

    # Calculer le chemin le plus court de ball_holder_index vers le but
    try:
        shortest_path = nx.shortest_path(G, source=ball_holder_index, target='goal', weight='weight')
    except nx.NetworkXNoPath:
        st.warning("No valid path to goal found.")
        return frame, G

    # Tracer le chemin le plus court en vert
    for k in range(len(shortest_path) - 1):
        start_pos = G.nodes[shortest_path[k]]['pos']
        end_pos = G.nodes[shortest_path[k + 1]]['pos']
        frame = cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)  # Tracez les lignes en vert

    return frame, G

def main():
    st.set_page_config(page_title="AI Powered Web Application for Football Tactical Analysis", layout="wide", initial_sidebar_state="expanded")
    st.title("Football Players Detection With Team Prediction & Tactical Map")
    st.subheader(":green[Works only with Tactical Camera footage]")

    st.sidebar.title("Main Settings")
    st.sidebar.markdown('---')
    st.sidebar.subheader("Video Upload")
    input_video_file = st.sidebar.file_uploader('Upload a video file', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if input_video_file:
        tempf.write(input_video_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()
        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)
    else:
        st.sidebar.warning('Please upload a video file to proceed.')

    # Load the YOLOv8 players detection model
    model_players = YOLO("../models/Yolo8L Players/weights/best.pt")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("../models/Yolo8M Field Keypoints/weights/best.pt")

    st.sidebar.markdown('---')
    st.sidebar.subheader("Team Names")
    team1_name = st.sidebar.text_input(label='First Team Name', value="")
    team2_name = st.sidebar.text_input(label='Second Team Name', value="")
    st.sidebar.markdown('---')

    ## Page Setup
    tab1, tab2, tab3, tab4 = st.tabs(["How to use?", "Team Colors", "Model Hyperparameters & Detection", "Recommend Pass"])

    with tab1:
        st.header(':green[Welcome!]')
        st.subheader('Main Application Functionalities:', divider='green')
        st.markdown("""
                    1. Football players, referee, and ball detection.
                    2. Players team prediction.
                    3. Estimation of players and ball positions on a tactical map.
                    4. Ball Tracking.
                    """)
        st.subheader('How to use?', divider='green')
        st.markdown("""
                    1. Upload a video to analyse, using the sidebar menu "Browse files" button.
                    2. Enter the team names that corresponds to the uploaded video in the text fields in the sidebar menu.
                    3. Access the "Team colors" tab in the main page.
                    4. Select a frame where players and goalkeepers from both teams can be detected.
                    5. Follow the instruction on the page to pick each team colors.
                    6. Go to the "Model Hyperparameters & Detection" tab, adjust hyperparameters and select the annotation options. (Default hyperparameters are recommended)
                    7. Run Detection!
                    8. If "save outputs" option was selected the saved video can be found in the "outputs" directory
                    """)
        st.write("Version 0.0.1")

    with tab2:
        if input_video_file:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to pick team colors from")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            if not success:
                st.error("Failed to read the video file. Please upload a valid video file.")
            else:
                with st.spinner('Detecting players in selected frame..'):
                    results = model_players(frame, conf=0.7)
                    bboxes = results[0].boxes.xyxy.cpu().numpy()
                    labels = results[0].boxes.cls.cpu().numpy()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections_imgs_list = []
                    detections_imgs_grid = []
                    padding_img = np.ones((80, 60, 3), dtype=np.uint8) * 255
                    for i, j in enumerate(list(labels)):
                        if int(j) == 0:
                            bbox = bboxes[i, :]
                            obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                            obj_img = cv2.resize(obj_img, (60, 80))
                            detections_imgs_list.append(obj_img)
                    detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list) // 2)])
                    detections_imgs_grid.append(
                        [detections_imgs_list[i] for i in range(len(detections_imgs_list) // 2, len(detections_imgs_list))])
                    if len(detections_imgs_list) % 2 != 0:
                        detections_imgs_grid[0].append(padding_img)
                    concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                    concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                    concat_det_imgs = cv2.vconcat([concat_det_imgs_row1, concat_det_imgs_row2])
                st.write("Detected players")
                value = streamlit_image_coordinates(concat_det_imgs, key="team_colors")
                st.markdown('---')
                radio_options = [f"{team1_name} P color", f"{team1_name} GK color", f"{team2_name} P color", f"{team2_name} GK color"]
                active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options,
                                        horizontal=True,
                                        help="Chose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
                if value is not None:
                    picked_color = concat_det_imgs[value['y'], value['x'], :]
                    st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)
                st.write("Boxes below can be used to manually adjust selected colors.")
                cp1, cp2, cp3, cp4 = st.columns([1, 1, 1, 1])
                with cp1:
                    hex_color_1 = st.session_state[f"{team1_name} P color"] if f"{team1_name} P color" in st.session_state else "#FFFFFF"
                    team1_p_color = st.color_picker(label=' ', value=hex_color_1, key='t1p')
                    st.session_state[f"{team1_name} P color"] = team1_p_color
                with cp2:
                    hex_color_2 = st.session_state[f"{team1_name} GK color"] if f"{team1_name} GK color" in st.session_state else "#FFFFFF"
                    team1_gk_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk')
                    st.session_state[f"{team1_name} GK color"] = team1_gk_color
                with cp3:
                    hex_color_3 = st.session_state[f"{team2_name} P color"] if f"{team2_name} P color" in st.session_state else "#FFFFFF"
                    team2_p_color = st.color_picker(label=' ', value=hex_color_3, key='t2p')
                    st.session_state[f"{team2_name} P color"] = team2_p_color
                with cp4:
                    hex_color_4 = st.session_state[f"{team2_name} GK color"] if f"{team2_name} GK color" in st.session_state else "#FFFFFF"
                    team2_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk')
                    st.session_state[f"{team2_name} GK color"] = team2_gk_color
            st.markdown('---')
            extracted_frame = st.empty()
            if success:
                extracted_frame.image(frame, use_column_width=True, channels="BGR")

    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state.get(f"{team1_name} P color", "#FFFFFF"), st.session_state.get(f"{team1_name} GK color", "#FFFFFF"),
                                                     team2_name, st.session_state.get(f"{team2_name} P color", "#FFFFFF"), st.session_state.get(f"{team2_name} GK color", "#FFFFFF"))

    with tab3:
        t2col1, t2col2 = st.columns([1, 1])
        with t2col1:
            player_model_conf_thresh = st.slider('Players Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.6, key='player_conf')
            keypoints_model_conf_thresh = st.slider('Field Keypoints Players Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.7, key='keypoint_conf')
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement RMSE Tolerance (pixels)', min_value=-1, max_value=100, value=7,
                                                         help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                           in current and previous detections. It is used to determine whether to update homography matrix or not. ", key='keypoint_tol')
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
                                    help="How many colors to extract from detected players bounding-boxes? It is used for team prediction.", key='num_colors')
            st.markdown("---")
            save_output = st.checkbox(label='Save output', value=False, key='save_output')
            if save_output:
                output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.', key='output_name')
            else:
                output_file_name = None
        st.markdown("---")

        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        with bcol1:
            nbr_frames_no_ball_thresh = st.number_input("Ball track reset threshold (frames)", min_value=1, max_value=10000,
                                                     value=30, help="After how many frames with no ball detection, should the track be reset?", key='ball_reset_thresh')
            ball_track_dist_thresh = st.number_input("Ball track distance threshold (pixels)", min_value=1, max_value=1280,
                                                        value=100, help="Maximum allowed distance between two consecutive balls detection to keep the current track.", key='ball_dist_thresh')
            max_track_length = st.number_input("Maximum ball track length (Nbr. detections)", min_value=1, max_value=1000,
                                                        value=35, help="Maximum total number of ball detections to keep in tracking history", key='track_length')
            ball_track_hyperparams = {
                0: nbr_frames_no_ball_thresh,
                1: ball_track_dist_thresh,
                2: max_track_length
            }
        with bcol2:
            st.write("Annotation options:")
            bcol21t, bcol22t = st.columns([1, 1])
            with bcol21t:
                show_k = st.checkbox(label="Show Keypoints Detections", value=False, key='show_k')
                show_p = st.checkbox(label="Show Players Detections", value=True, key='show_p')
            with bcol22t:
                show_pal = st.checkbox(label="Show Color Palettes", value=True, key='show_pal')
                show_b = st.checkbox(label="Show Ball Tracks", value=True, key='show_b')
            plot_hyperparams = {
                0: show_k,
                1: show_pal,
                2: show_b,
                3: show_p
            }
        with bcol3:
            st.write('')

        bcol31, bcol32, bcol33 = st.columns([1.5, 1, 1])
        with bcol31:
            st.write('')
        with bcol32:
            start_detection = st.button(label='Start Detection', disabled=not input_video_file, key='start_detection')
        with bcol33:
            stop_btn_state = st.session_state.get('detection_running', False)
            stop_detection = st.button(label='Stop Detection', disabled=not stop_btn_state, key='stop_detection')

        if start_detection:
            st.toast("Starting detection...", icon="🔄")
            st.session_state['detection_running'] = True
            stframe = st.empty()
            cap = cv2.VideoCapture(tempf.name)
            homog = detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
                            detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
                            num_pal_colors, colors_dic, color_list_lab)

            if homog is not None:
                # Définir les positions des joueurs et de l'holder du ballon
                player_positions = []
                player_team_indices = []
                ball_holder_center = None

                for i, label in enumerate(labels):
                    if label == 0:  # Joueur
                        bbox = bboxes[i]
                        player_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                        player_positions.append(player_center)
                        player_team_indices.append(labels[i])
                    elif label == 2:  # Ballon
                        bbox = bboxes[i]
                        ball_holder_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

                # Transformation des coordonnées
                player_positions_on_map = transform_points(player_positions, homog)
                ball_holder_position_on_map = transform_points([ball_holder_center], homog)[0]

                # Stocker les résultats dans st.session_state
                st.session_state['homog'] = homog
                st.session_state['player_positions'] = player_positions_on_map
                st.session_state['ball_holder_position'] = ball_holder_position_on_map
                st.session_state['player_team_indices'] = player_team_indices

                # Indiquer que la détection est terminée
                st.session_state['detection_done'] = True

                st.toast(f'Detection Completed!', icon="✅")
                cap.release()
            else:
                st.error('Detection failed.')
                st.toast(f'Detection failed.', icon="⚠️")
                st.session_state['detection_done'] = False

    with tab4:
        if 'detection_done' in st.session_state and st.session_state['detection_done']:
            # Vérifiez que player_positions est défini
            if 'player_positions' not in st.session_state:
                st.error("Player positions not found. Please run detection first.")
                return

            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select frame for recommendation", min_value=1, max_value=frame_count, step=1, help="Select frame to recommend passes from")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            if not success:
                st.error("Failed to read the video file.")
            else:
                stframe = st.empty()
                stframe.image(frame, use_column_width=True, channels="BGR")

                if st.button('Recommend Pass', key='recommend_pass'):
                    st.toast("Starting pass recommendation...", icon="🔄")

                    ball_holder_index = st.session_state['player_team_indices'].index(0)  # Assurer que le détenteur du ballon est trouvé
                    
                    # Assumer que le but de l'équipe adverse est à gauche ou à droite du terrain, selon le cadre de la vidéo
                    goal_position = (0, frame.shape[0] // 2)  # Exemple : but à gauche
                    # goal_position = (frame.shape[1], frame.shape[0] // 2)  # Exemple : but à droite
                    
                    opponent_positions = [st.session_state['player_positions'][i] for i in range(len(st.session_state['player_positions'])) if st.session_state['player_team_indices'][i] != st.session_state['player_team_indices'][ball_holder_index]]
                    
                    frame_with_shortest_path, G = plot_shortest_path_to_goal(
                        st.session_state['player_positions'], 
                        st.session_state['player_team_indices'], 
                        ball_holder_index, 
                        goal_position, 
                        frame,
                        opponent_positions
                    )
                    
                    st.image(frame_with_shortest_path, channels="BGR", use_column_width=True)
                    st.toast("Pass recommendation completed successfully!", icon="✅")
        else:
            st.warning("Please run detection first to use this feature.")
            st.toast("Detection must be completed before selecting frames for recommendation.", icon="⚠️")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
