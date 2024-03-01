import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
from moviepy.editor import VideoFileClip
import librosa
from whisperTS.main import create_sentences_from_audio
from video import *
from utils import *

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Constants
HOST = "0.0.0.0"
PORT = 8080
GENERATING = False
LOCALVIDDIR = "../temp/"
COUNT = 0

@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Handles the POST request to the `/api/generate` endpoint. It processes the provided YouTube video URL by downloading the video, detecting and trimming silent segments, and optionally adding subtitles based on the audio content.

    Args:
        None directly; utilizes global variables and JSON payload from the POST request containing keys: 'aiModel', 'videoUrl', 'strokeColor', 'fontOutline', 'fontSize'.

    Returns:
        flask.Response: A JSON response indicating the status ('success' or 'error') and a message, which in the case of success includes the path to the processed video.
    """
    try:
          # Set global variable
        global GENERATING
        GENERATING = True
        COUNT+1
        
        # Clean
        clean_dir(LOCALVIDDIR)
        clean_dir("../subtitles/")
 
        # Parse JSON
        data = request.get_json()
        ai_model = data.get('aiModel')  # Get the AI model selected by the user
        video_url = data.get('videoUrl') # URL of Video -> Download Video to temp
        strokeColor = data.get('strokeColor') # URL of Video -> Download Video to temp
        fontOutline = data.get('fontOutline') # URL of Video -> Download Video to temp
        fontSize = data.get('fontSize') # URL of Video -> Download Video to temp
        print(colored(f"[+] Downloading with YTDLP", "blue"))

        video_clips = []
        if video_url:
            # Process the provided YouTube video
            ydl_opts = {
                'format': 'best',
                'outtmpl': '../temp/%(id)s.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'keepvideo': True,  # Keep the video file after extracting audio
         }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get('id', None)
            audio_path = f"../temp/{video_id}.mp3"
            video_path = f"../temp/{video_id}.mp4"
            
        try:
            print(colored(f"[+] Downloaded with YTDLP and {data}", "green"))
            print(colored(f"[+] Identifying Silences from Video", "blue"))
            silent_segments = detect_silent_segments(audio_path)

        except PermissionError as e:
            print(f"Permission error when accessing {audio_path}. Retrying...")
            time.sleep(20)  # Wait a bit for the file to be released
            silent_segments = detect_silent_segments(audio_path)
        
        try:
            print(colored(f"[+] Creating Clips from video", "blue"))
            video_clips = trim_silences_from_video(video_path, silent_segments, strokeColor, fontOutline, fontSize)
            combined_video_path = combine_videos(video_clips, 500, 160)
      
        except PermissionError:
            print(colored(f"Error when rendering {video_clips}...", "red"))
            return jsonify({"status": "error", "message": str(e)})
        
        return jsonify({"status": "success", "message": "Video processed successfully.", "clips": combined_video_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def detect_silent_segments(audio_path):
    """
    Detects silent segments within an audio file by analyzing its volume levels.
    
    Args:
        audio_path (str): Path to the audio file to analyze for silent segments.
    
    Returns:
        List[Tuple[float, float]]: A list of tuples, each representing the start and end times (in seconds) of silent segments within the audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)
    non_silent_segments = librosa.effects.split(y, top_db=30)
    silent_segments = invert_segments(non_silent_segments, len(y), sr)
    return silent_segments


def invert_segments(segments, y_length, sr):
    """
    Inverts non-silent segments to identify silent segments in an audio track. This is useful for finding periods of silence between detected sounds.
    
    Args:
        segments (List[Tuple[int, int]]): A list of tuples representing the non-silent segments in the audio file.
        y_length (int): The total number of frames in the audio file.
        sr (int): The sampling rate of the audio file.
    
    Returns:
        List[Tuple[float, float]]: A list of tuples, each indicating the start and end times (in seconds) of silent segments.
    """

    silent_segments = []
    start = 0
    for seg in segments:
        silent_start = librosa.frames_to_time(start, sr=sr)
        silent_end = librosa.frames_to_time(seg[0], sr=sr)
        silent_segments.append((silent_start, silent_end))
        start = seg[1]
    if start < y_length:
        silent_end = librosa.frames_to_time(y_length, sr=sr)
        silent_segments.append((librosa.frames_to_time(start, sr=sr), silent_end))
    return silent_segments


def createClip(video_path, start, end, name, strokeColor, fontOutline, fontSize):
    print(colored(f"[+] Creating Clip using AI", "blue"))
    """
    Creates a video clip from the specified time segment, applies subtitles, and saves the clip to a file.

    Args:
        video_path (str): Path to the source video file.
        start (float): The start time of the clip in the video (in seconds).
        end (float): The end time of the clip in the video (in seconds).
        name (str): A unique identifier for the output file.
        strokeColor (str): Color for the stroke of the subtitle text.
        fontOutline (str): Color for the font outline.
        fontSize (int): Font size for the subtitles.

    Returns:
        str: The path to the output video file with the applied subtitles.
    """

    try: 
        # Define Stuff
        clip_path = f"{LOCALVIDDIR}{name}.mp4"
        captioned_clip_path = f"{LOCALVIDDIR}captioned_{name}.mp4"
        audio_path = f"{LOCALVIDDIR}{name}.mp3"
        audio_clip = []
        # Save Clip
        video = VideoFileClip(video_path)
        video.subclip(start, end).write_videofile(clip_path, fps=30, codec="libx264")
        video.subclip(start, end).audio.write_audiofile(audio_path)
        # Generate Pieces
        audio_clip.append(video.subclip(start, end).audio)
        stt = (create_sentences_from_audio(audio_path)).split(". ")
        stt = list(filter(lambda x: x != "", stt))
        print(colored(f"[+] Creating subtitles with\n {audio_path}\n {stt}\n {audio_clip}\n", "blue"))
        subtitles_path = generate_subtitles(
            audio_path=audio_path, 
            sentences=stt,
            audio_clips=audio_clip,
            )
        print(colored(f"[+] Creating captions", "blue"))
        # Combine Clip Pieces
        generator = lambda txt: TextClip(
            txt,
            font="../fonts/bold_font.ttf",
            fontsize=fontSize,
            color=fontOutline,
            stroke_color=strokeColor,
            stroke_width=5,
        )

        # Burn the subtitles into the video
        subtitles = SubtitlesClip(subtitles_path, generator)
        result = CompositeVideoClip([
            VideoFileClip(clip_path),
            subtitles.set_pos(("center", "center"))
        ])

        # Add the audio
        audio = AudioFileClip(audio_path)
        result = result.set_audio(audio)
        result.write_videofile(captioned_clip_path, threads=2)
        print(colored(f"[+] Created Clip using AI", "green"))
        return captioned_clip_path
    except Exception as e:
        return
    

def trim_silences_from_video(video_path, silent_segments, strokeColor, fontOutline, fontSize):
    """
    Trims silent segments from the video based on the provided silent segments list and generates a series of video clips without these silent parts. It also applies subtitles to these clips if necessary.

    Args:
        video_path (str): Path to the source video file.
        silent_segments (List[Tuple[float, float]]): List of start and end times (in seconds) of silent segments to trim from the video.
        strokeColor (str), fontOutline (str), fontSize (int): Styling parameters for the subtitles.

    Returns:
        List[str]: A list of paths to the generated video clips without silent segments, with subtitles applied if specified.
    """

    video = VideoFileClip(video_path)
    video_duration = video.duration  # Get the duration of the video
    clips = []
    name = len(clips)
    for start, end in silent_segments:
        # Ensure the start and end times are within the video's duration
        if start < video_duration and end < video_duration:
            final_video_path = createClip(
                video_path=video_path, 
                start=start, 
                end=end, 
                name=name,
                strokeColor=strokeColor, 
                fontOutline=fontOutline, 
                fontSize=fontSize, 
                )
        elif end >= video_duration:
            final_video_path = createClip(
                video_path=video_path, 
                start=start, 
                end=(end - 1), 
                name=name,
                strokeColor=strokeColor, 
                fontOutline=fontOutline, 
                fontSize=fontSize,             
                    )
            if len(final_video_path):
                clips.append(final_video_path)
            return
    video.close()  # Close the original video clip
    return clips

if __name__ == '__main__':
   # Run Flask App
    app.run(debug=True, host=HOST, port=PORT)
