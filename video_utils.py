import os
import json
import ffmpeg
import logging
from datetime import datetime
import subprocess
import uuid
import shutil

logger = logging.getLogger(__name__)

def get_video_duration(file_path):
    """
    Get the duration of a video file in seconds using ffmpeg.
    Returns None if there's an error.
    """
    try:
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.error(f"Error getting video duration for {file_path}: {str(e)}")
        return None

def create_video_thumbnail(video_path, output_path, time_position=5):
    """
    Create a thumbnail from a video at the specified time position.
    """
    try:
        (
            ffmpeg
            .input(video_path, ss=time_position)
            .output(output_path, vframes=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except Exception as e:
        logger.error(f"Error creating thumbnail for {video_path}: {str(e)}")
        return False

def segment_video(video_path, output_pattern, segment_time=10):
    """
    Segment a video into chunks of a specified duration.
    output_pattern should be something like "output_dir/video_%03d.mp4"
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                output_pattern,
                c='copy',  # Use the same codecs
                map='0',   # Use all streams
                f='segment',
                segment_time=segment_time
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except Exception as e:
        logger.error(f"Error segmenting video {video_path}: {str(e)}")
        return False

def get_segment_for_timestamp(base_path, segment_duration, timestamp):
    """
    Calculate which segment file contains the given timestamp.
    
    Args:
        base_path (str): Base path without the segment number
        segment_duration (int): Duration of each segment in seconds
        timestamp (float): Timestamp in seconds from the start of the video
        
    Returns:
        tuple: (segment_path, position_in_segment)
    """
    segment_number = int(timestamp // segment_duration)
    position_in_segment = timestamp % segment_duration
    
    # Assuming the segment filenames follow a pattern like "video_000.mp4"
    segment_path = f"{base_path}_{segment_number:03d}.mp4"
    
    return segment_path, position_in_segment

def calculate_bitrate_for_bandwidth(available_bandwidth, safety_factor=0.8):
    """
    Calculate an appropriate bitrate based on available bandwidth.
    
    Args:
        available_bandwidth (int): Available bandwidth in bits per second
        safety_factor (float): Factor to ensure we don't use the full bandwidth
        
    Returns:
        int: Calculated bitrate in bits per second
    """
    # Apply safety factor to avoid using 100% of bandwidth
    safe_bandwidth = available_bandwidth * safety_factor
    
    # Round to nearest standard bitrate
    standard_bitrates = [500000, 1000000, 2500000, 5000000, 8000000, 12000000]
    
    for bitrate in standard_bitrates:
        if bitrate <= safe_bandwidth:
            selected_bitrate = bitrate
        else:
            break
            
    return selected_bitrate

def create_adaptive_streams(input_video, output_dir, bitrates=None):
    """
    Create multiple renditions of a video for adaptive streaming.
    
    Args:
        input_video (str): Path to input video
        output_dir (str): Directory to store output files
        bitrates (list): List of bitrates to generate, defaults to standard set
        
    Returns:
        list: Paths to created renditions
    """
    if bitrates is None:
        # Standard bitrates for different quality levels (bits/second)
        bitrates = [
            {"bitrate": 500000, "resolution": "640x360"},   # Low
            {"bitrate": 1500000, "resolution": "960x540"},  # Medium
            {"bitrate": 3000000, "resolution": "1280x720"}, # High
            {"bitrate": 6000000, "resolution": "1920x1080"} # Full HD
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    
    try:
        # Get input video info
        probe = ffmpeg.probe(input_video)
        
        for profile in bitrates:
            output_file = os.path.join(
                output_dir, 
                f"rendition_{profile['resolution'].replace('x', '_')}_{profile['bitrate']//1000}k.mp4"
            )
            
            # Create the rendition
            (
                ffmpeg
                .input(input_video)
                .output(
                    output_file,
                    vcodec='libx264',
                    acodec='aac',
                    video_bitrate=profile['bitrate'],
                    audio_bitrate='128k',
                    s=profile['resolution']
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            output_files.append(output_file)
            
        return output_files
    
    except Exception as e:
        logger.error(f"Error creating adaptive streams for {input_video}: {str(e)}")
        return []

def get_video_metadata(video_path):
    """
    Extract metadata from a video file using FFmpeg
    """
    try:
        # Run FFprobe to get video information in JSON format
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Extract video stream data
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            logger.warning(f"No video stream found in {video_path}")
            return {
                "duration": 0,
                "width": 0,
                "height": 0,
                "codec": "unknown",
                "bitrate": 0
            }
        
        # Extract format data
        format_data = data.get('format', {})
        
        # Get duration
        duration = float(format_data.get('duration', 0))
        
        # Get dimensions
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        
        # Get codec
        codec = video_stream.get('codec_name', 'unknown')
        
        # Get bitrate
        bitrate_str = format_data.get('bit_rate', '0')
        try:
            bitrate = int(bitrate_str) // 1000  # Convert to kbps
        except (ValueError, TypeError):
            bitrate = 0
        
        return {
            "duration": duration,
            "width": width,
            "height": height,
            "codec": codec,
            "bitrate": bitrate
        }
    
    except Exception as e:
        logger.error(f"Error extracting metadata from {video_path}: {e}")
        return {
            "duration": 0,
            "width": 0,
            "height": 0,
            "codec": "unknown",
            "bitrate": 0
        }

def create_video_thumbnails(video_path, output_dir, video_id, num_thumbnails=3):
    """
    Create thumbnails for a video at different timestamps
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration
        metadata = get_video_metadata(video_path)
        duration = metadata['duration']
        
        if duration <= 0:
            logger.warning(f"Invalid duration for {video_path}")
            return []
        
        thumbnail_paths = []
        
        # Create thumbnails at different positions
        for i in range(num_thumbnails):
            # Calculate position (evenly distributed)
            position = duration * (i + 1) / (num_thumbnails + 1)
            
            # Output path for this thumbnail
            thumbnail_path = os.path.join(output_dir, f"{video_id}_thumb_{i}.jpg")
            
            # FFmpeg command to create thumbnail
            cmd = [
                'ffmpeg',
                '-ss', str(position),
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2',
                '-y',
                thumbnail_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            thumbnail_paths.append(thumbnail_path)
            logger.info(f"Created thumbnail at position {position:.2f}s: {thumbnail_path}")
        
        return thumbnail_paths
    
    except Exception as e:
        logger.error(f"Error creating thumbnails for {video_path}: {e}")
        return []

def segment_video_for_streaming(video_path, output_dir, video_id):
    """
    Segment a video into HLS chunks
    """
    try:
        # Create directory for this video's segments
        video_output_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Path for the playlist file
        playlist_path = os.path.join(video_output_dir, "playlist.m3u8")
        
        # FFmpeg command to create HLS segments
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-start_number', '0',
            '-hls_time', '10',
            '-hls_list_size', '0',
            '-f', 'hls',
            playlist_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Created HLS segments for {video_path} at {playlist_path}")
        
        return playlist_path
    
    except Exception as e:
        logger.error(f"Error segmenting video {video_path}: {e}")
        return None

def create_adaptive_stream(video_path, output_dir, video_id):
    """
    Create an adaptive bitrate stream with multiple quality levels
    """
    try:
        # Create directory for this video's streams
        video_output_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Master playlist path
        master_path = os.path.join(video_output_dir, "master.m3u8")
        
        # Get video metadata
        metadata = get_video_metadata(video_path)
        width = metadata['width']
        height = metadata['height']
        
        # Determine resolutions based on original video
        resolutions = []
        
        # Always include original resolution
        resolutions.append((width, height))
        
        # Add lower resolutions if the original is high enough
        if width >= 1280 and height >= 720:
            resolutions.append((854, 480))  # 480p
            resolutions.append((640, 360))  # 360p
        elif width >= 854 and height >= 480:
            resolutions.append((640, 360))  # 360p
        
        # Create the master playlist file
        with open(master_path, 'w') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            
            # For each resolution, create a variant stream
            for i, (w, h) in enumerate(resolutions):
                # Calculate bandwidth (bitrate) based on resolution
                # This is an estimate - in a real system you'd want to measure actual bitrate
                bandwidth = int(w * h * 0.1)  # Simple heuristic
                
                # Variant playlist name
                variant_name = f"stream_{w}x{h}.m3u8"
                variant_path = os.path.join(video_output_dir, variant_name)
                
                # Add to master playlist
                f.write(f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={w}x{h}\n")
                f.write(f"{variant_name}\n")
                
                # Create the variant stream
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-vf', f'scale={w}:{h}',
                    '-b:v', f'{bandwidth//1000}k',
                    '-maxrate', f'{int(bandwidth*1.5)//1000}k',
                    '-bufsize', f'{bandwidth//1000}k',
                    '-hls_time', '10',
                    '-hls_list_size', '0',
                    '-hls_segment_filename', os.path.join(video_output_dir, f"segment_{w}x{h}_%03d.ts"),
                    '-f', 'hls',
                    '-y',
                    variant_path
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                logger.info(f"Created variant stream at {w}x{h}: {variant_path}")
        
        logger.info(f"Created adaptive HLS stream for {video_path} at {master_path}")
        return master_path
    
    except Exception as e:
        logger.error(f"Error creating adaptive stream for {video_path}: {e}")
        return None 