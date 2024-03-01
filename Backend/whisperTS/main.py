import whisper_timestamped as whisper
import os
from moviepy.editor import VideoFileClip
import whisper

def create_sentences_from_audio(audioFile):
    """
    Generates sentences / captions from a given audio file using whisper ai and returns the result

    Args:
        audioFile (str): The path to the audio file to generate sentence from.

    Returns:
        result: The text generated.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audioFile,fp16=False)
    return result["text"]

def convert_video_to_audio_moviepy(video_file, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename= os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename[0]}.{output_ext}") 
    return f"{filename}.{output_ext}"

def find_segment_silences(segments):
    
    silences = []
    last_end_time = segments[0]['end'] if segments else 0  # Start with the end of the first segment

    for segment in segments[1:]:  # Start from the second segment
        current_start_time = segment['start']
        # If there's a gap between the end of the last segment and the start of the current one
        if current_start_time - last_end_time > 0:
            silences.append({
                'start': last_end_time,
                'end': current_start_time
            })
        last_end_time = segment['end']  # Update the end time for the next iteration

    return silences

def create_transcript_from_audio(audio_file):
    """
    Generates sentences / captions from a given audio file using whisper ai timestamped and returns the result

    Args:
        audioFile (str): The path to the audio file to generate sentence from.

    Returns:
        result: The text generated.
    """
    
    # Load the new audio file
    audio = whisper.load_audio(audio_file)

    # Load the Whisper model
    model = whisper.load_model("tiny", device="cpu")

    # Transcribe the audio with the specified language
    result = whisper.transcribe(model, audio, language="en")
    print(result)
    return result