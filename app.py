from flask import Flask, request, jsonify, Response, send_file, url_for
from flask_cors import CORS
import os
import json
import time
from datetime import datetime, timedelta
import threading
import logging
import uuid
import shutil
from video_utils import get_video_metadata, create_video_thumbnails, create_adaptive_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure streaming performance options
STREAMING_CONFIG = {
    "cache_ttl": 3600,  # Cache time for static segments (1 hour)
    "segment_length": 2,  # 2-second segments for lower latency
    "buffer_profile": {
        "short": {"ahead": 10, "sync_interval": 1200, "prefetch": 3},
        "medium": {"ahead": 20, "sync_interval": 2000, "prefetch": 5},
        "long": {"ahead": 30, "sync_interval": 3000, "prefetch": 7}
    },
    "bandwidth_profiles": [
        {"bitrate": 500000, "resolution": "640x360", "segment_size": "250KB"},
        {"bitrate": 1500000, "resolution": "960x540", "segment_size": "750KB"},
        {"bitrate": 3000000, "resolution": "1280x720", "segment_size": "1.5MB"},
        {"bitrate": 6000000, "resolution": "1920x1080", "segment_size": "3MB"}
    ],
    # Multi-stream settings
    "max_concurrent_streams": 10,  # Maximum number of concurrent streams
    "stream_switch_buffer": 5  # Seconds to buffer when switching streams
}

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Add CDN-friendly cache headers
@app.after_request
def add_header(response):
    # Add cache headers for HLS segments
    if request.path.endswith('.ts') or request.path.endswith('.m3u8'):
        response.headers['Cache-Control'] = f'public, max-age={STREAMING_CONFIG["cache_ttl"]}'
        response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Directory where video files are stored
VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Store user preferences and streams
USER_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
VIDEO_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos.json")

# Global variables
global_start_time = None  # Will be set when we start the global streaming
user_streams = {}  # User ID -> List of video IDs in order
video_metadata = {}  # Video ID -> {duration, path, name, etc.}
active_streams = {}  # Stream ID -> {owner, start_time, video_ids, viewers}

# Helper functions
def load_data():
    global user_streams, video_metadata
    
    # Load user streams if file exists
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            user_streams = json.load(f)
    
    # Load video metadata if file exists
    if os.path.exists(VIDEO_DB_FILE):
        with open(VIDEO_DB_FILE, 'r') as f:
            video_metadata = json.load(f)
    
    # Validate that video files actually exist and remove any that don't
    videos_to_remove = []
    for video_id, video_info in video_metadata.items():
        video_path = os.path.join(VIDEOS_DIR, video_info.get("path", ""))
        if not os.path.exists(video_path):
            videos_to_remove.append(video_id)
    
    # Remove videos that don't exist from metadata
    for video_id in videos_to_remove:
        logger.info(f"Removing non-existent video from metadata: {video_id}")
        del video_metadata[video_id]
    
    # If we removed any videos, save the updated metadata
    if videos_to_remove:
        save_video_data()
        
        # Also remove these videos from user streams
        for user_id in user_streams:
            user_streams[user_id]["videos"] = [v for v in user_streams[user_id].get("videos", []) if v not in videos_to_remove]
        save_user_data()

def save_user_data():
    with open(USER_DB_FILE, 'w') as f:
        json.dump(user_streams, f, indent=2)

def save_video_data():
    with open(VIDEO_DB_FILE, 'w') as f:
        json.dump(video_metadata, f, indent=2)

# Load data at startup
load_data()

# API Endpoints
@app.route('/api/start-global-stream', methods=['POST'])
def start_global_stream():
    global global_start_time
    
    # Check if we have any videos available
    if not video_metadata:
        logger.error("Stream start failed: No videos available")
        return jsonify({"error": "No videos available to stream. Please upload videos first."}), 400
    
    # Set the global start time to current time
    global_start_time = datetime.now().isoformat()
    logger.info(f"Global stream started at {global_start_time}")
    
    # Store the start time in a file for persistence
    start_time_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_start_time.txt")
    with open(start_time_path, 'w') as f:
        f.write(global_start_time)
    logger.info(f"Global start time saved to {start_time_path}")
    
    # Make sure all users have the same video sequence
    available_videos = list(video_metadata.keys())
    logger.info(f"Available videos for stream: {available_videos}")
    
    # Count of users updated
    users_updated = 0
    
    for user_id in user_streams:
        # Ensure each user has exactly the uploaded videos, in the same order
        user_streams[user_id]["videos"] = available_videos.copy()
        users_updated += 1
    
    logger.info(f"Updated video sequence for {users_updated} users")
    save_user_data()
    
    return jsonify({"status": "success", "start_time": global_start_time})

@app.route('/api/register-user', methods=['POST'])
def register_user():
    data = request.json
    user_id = str(uuid.uuid4())
    
    # Store user preferences (can be extended)
    user_streams[user_id] = {
        "preferences": data.get("preferences", {}),
        "videos": []  # Will be filled with videos later
    }
    
    # Add all existing videos to the user's stream
    available_videos = list(video_metadata.keys())
    if available_videos:
        user_streams[user_id]["videos"] = available_videos
    
    save_user_data()
    
    return jsonify({"user_id": user_id})

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    data = request.get_json()
    title = data.get('title', 'Untitled Video')
    
    # Generate a unique ID for the video
    video_id = str(uuid.uuid4())
    
    # Store video metadata with name and upload time
    current_time = int(time.time())
    formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
    
    video_metadata[video_id] = {
        'id': video_id,
        'title': title,
        'duration': 0,  # Will be updated when the video is processed
        'upload_time': current_time,
        'formatted_upload_time': formatted_time,
        'processed': False,
        'hls_url': f'/api/stream-hls/{video_id}/master.m3u8',
        'thumbnail_url': f'/api/thumbnails/{video_id}/thumbnail.jpg'
    }
    
    # Update all user streams to include this video
    for user_id in user_streams:
        if 'videos' in user_streams[user_id]:
            user_streams[user_id]['videos'].append(video_id)
    
    # Save the updated metadata
    save_video_data()
    save_user_data()
    
    return jsonify({
        'success': True,
        'videoId': video_id,
        'message': 'Video registered successfully'
    })

@app.route('/api/upload-video-file', methods=['POST'])
def upload_video_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        # Generate a safe filename
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(VIDEOS_DIR, filename)
        
        try:
            # Create videos directory if it doesn't exist
            os.makedirs(VIDEOS_DIR, exist_ok=True)
            
            # Save the file
            file.save(filepath)
            logger.info(f"Saved uploaded file to {filepath}")
            
            # Generate a unique video ID
            video_id = str(uuid.uuid4())
            
            # Get the name from the form or use the original filename without extension
            video_name = request.form.get('name', os.path.splitext(file.filename)[0])
            
            # Get video metadata using FFmpeg
            metadata = get_video_metadata(filepath)
            logger.info(f"Extracted metadata: {metadata}")
            
            # Create video thumbnails
            thumbnails_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnails")
            os.makedirs(thumbnails_dir, exist_ok=True)
            thumbnails = create_video_thumbnails(filepath, thumbnails_dir, video_id)
            
            # Create HLS stream segments for adaptive streaming
            streams_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streams")
            os.makedirs(streams_dir, exist_ok=True)
            video_streams_dir = os.path.join(streams_dir, video_id)
            os.makedirs(video_streams_dir, exist_ok=True)
            master_playlist = create_adaptive_stream(filepath, streams_dir, video_id)
            
            # Add to video metadata
            video_metadata[video_id] = {
                "name": video_name,
                "duration": metadata["duration"],
                "width": metadata["width"],
                "height": metadata["height"],
                "codec": metadata["codec"],
                "bitrate": metadata.get("bitrate", 0),
                "path": filename,
                "thumbnails": thumbnails,
                "master_playlist": master_playlist,
                "upload_date": datetime.now().isoformat(),
                "original_filename": file.filename
            }
            save_video_data()
            
            # Update all user streams to include the new video
            for user_id in user_streams:
                if video_id not in user_streams[user_id]["videos"]:
                    user_streams[user_id]["videos"].append(video_id)
            save_user_data()
            
            # Return success response
            return jsonify({
                "status": "success", 
                "message": "File uploaded and registered successfully",
                "filename": filename,
                "video_id": video_id,
                "metadata": video_metadata[video_id]
            })
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return jsonify({"error": f"Error processing video: {str(e)}"}), 500
    
    return jsonify({"error": "Unknown error"}), 500

@app.route('/api/get-current-video/<user_id>', methods=['GET'])
def get_current_video(user_id):
    if user_id not in user_streams:
        logger.warning(f"Unknown user requested video: {user_id}")
        return jsonify({
            "error": "User not found", 
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 404
    
    if not global_start_time:
        logger.warning("Video request when global stream not running")
        return jsonify({
            "error": "Global stream has not started", 
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 400
    
    # Calculate elapsed time since stream started with microsecond precision
    try:
        start_time = datetime.fromisoformat(global_start_time)
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        
        # Track precise timing for debugging
        logger.info(f"Video request: User={user_id}, Stream running for {elapsed_seconds:.6f} seconds")
        
        # Find which video should be playing based on elapsed time
        user_video_sequence = user_streams[user_id]["videos"]
        
        if not user_video_sequence:
            logger.warning(f"No videos in sequence for user {user_id}")
            return jsonify({
                "error": "No videos in user sequence",
                "server_time": int(current_time.timestamp() * 1000)
            }), 404
        
        # Calculate total playlist duration for looping
        total_duration = 0
        valid_videos = []
        video_start_times = {}  # Track when each video starts in the sequence
        cumulative_time = 0
        
        for video_id in user_video_sequence:
            if video_id in video_metadata:
                metadata = video_metadata.get(video_id, {})
                duration = metadata.get("duration", 0)
                if duration > 0:
                    # Record the start time of this video in the sequence
                    video_start_times[video_id] = cumulative_time
                    
                    total_duration += duration
                    cumulative_time += duration
                    valid_videos.append({
                        "id": video_id,
                        "duration": duration,
                        "title": metadata.get("name", "Unknown"),
                        "hls_enabled": metadata.get("hls_url") is not None,
                        "hls_url": metadata.get("hls_url"),
                        "thumbnail_url": metadata.get("thumbnail_url"),
                        "start_time_in_sequence": video_start_times[video_id]
                    })
                else:
                    logger.warning(f"Video {video_id} has invalid duration: {duration}")
            else:
                logger.warning(f"Video {video_id} not found in metadata")
        
        # Handle empty or invalid playlist
        if total_duration <= 0 or not valid_videos:
            logger.warning(f"No valid videos in playlist for user {user_id}")
            return jsonify({
                "error": "No valid videos in sequence",
                "server_time": int(current_time.timestamp() * 1000)
            }), 404
        
        # Calculate position in looping playlist with microsecond precision for perfect sync
        looped_position = elapsed_seconds % total_duration if total_duration > 0 else 0
        
        # Find the current video in the sequence
        position_tracker = 0
        current_video = None
        position_in_video = 0
        next_video = None
        
        for video in valid_videos:
            video_duration = video["duration"]
            
            if position_tracker + video_duration > looped_position:
                # This is the current video
                current_video = video["id"]
                position_in_video = looped_position - position_tracker
                
                # Determine next video for preloading
                next_idx = (valid_videos.index(video) + 1) % len(valid_videos)
                next_video = valid_videos[next_idx]["id"]
                
                # Calculate time until next video transition
                time_until_next = video_duration - position_in_video
                
                # Get current video metadata
                current_video_data = video_metadata.get(current_video, {})
                
                # Calculate precise server time for synchronization
                precise_server_time = int(current_time.timestamp() * 1000)
                precise_start_time = int(start_time.timestamp() * 1000)
                
                # Optimize buffer settings based on video type and position
                # Select profile based on video duration
                buffer_profile = "medium"
                if video_duration < 60:
                    buffer_profile = "short"
                elif video_duration > 300:
                    buffer_profile = "long"
                
                # Get buffer settings from the selected profile
                buffer_config = STREAMING_CONFIG["buffer_profile"][buffer_profile]
                buffer_ahead_time = buffer_config["ahead"]
                sync_interval = buffer_config["sync_interval"]
                prefetch_count = buffer_config["prefetch"]
                
                # Calculate player performance hint based on position in video
                position_ratio = position_in_video / video_duration if video_duration > 0 else 0
                performance_mode = "smooth"
                
                # Special cases that need more aggressive buffering:
                # 1. Near the beginning (first 10%)
                # 2. Near a transition (last 10 seconds)
                # 3. If network conditions indicate low bandwidth
                if position_ratio < 0.1 or time_until_next < 10:
                    performance_mode = "aggressive"
                    # Increase buffer for transition points
                    buffer_ahead_time = max(buffer_ahead_time, 20)
                
                # Calculate the exact keyframe time for perfect seeking
                keyframe_position = position_in_video
                # If we know the keyframe intervals, we could find the nearest one
                # For now we'll use the exact position and let the client handle it
                
                # Prepare segment preload hints for the client
                # This helps the client preload the correct segments to avoid buffering
                preload_segments = []
                
                # If using HLS, calculate segment URLs that should be preloaded
                if current_video_data.get("hls_url"):
                    base_segment_path = f"/api/stream-hls/{current_video}/segment"
                    current_segment = int(position_in_video / STREAMING_CONFIG["segment_length"])
                    
                    # Add the next few segments as preload hints
                    for i in range(current_segment + 1, current_segment + prefetch_count + 1):
                        segment_time = i * STREAMING_CONFIG["segment_length"]
                        if segment_time < video_duration:
                            preload_segments.append({
                                "url": f"{base_segment_path}_{i}.ts",
                                "time_offset": segment_time - position_in_video
                            })
                
                # If we're close to transitioning to the next video, add its initial segments too
                if time_until_next < 10 and next_video:
                    base_segment_path = f"/api/stream-hls/{next_video}/segment"
                    for i in range(0, 3):  # Preload first 3 segments of next video
                        preload_segments.append({
                            "url": f"{base_segment_path}_{i}.ts",
                            "time_offset": time_until_next + (i * STREAMING_CONFIG["segment_length"])
                        })
                
                logger.info(f"Returning video: {current_video}, position: {position_in_video:.6f}s, buffer: {buffer_ahead_time}s")
                
                return jsonify({
                    "video_id": current_video,
                    "id": current_video,
                    "title": current_video_data.get("name", "Unknown"),
                    "name": current_video_data.get("name", "Unknown"),
                    "position": position_in_video,
                    "keyframe_position": keyframe_position,  # Precise position for seeking
                    "duration": video_duration,
                    "elapsed_total": elapsed_seconds,
                    "next_video": next_video,
                    "time_until_next": time_until_next,
                    "server_time": precise_server_time,
                    "playlist_duration": total_duration,
                    "playlist_position": looped_position,
                    "stream_start_time": precise_start_time,
                    "sync_interval": sync_interval,
                    "hls_enabled": current_video_data.get("hls_url") is not None,
                    "hls_url": current_video_data.get("hls_url"),
                    "thumbnail_url": current_video_data.get("thumbnail_url"),
                    "force_sync": True,
                    "buffer_ahead_time": buffer_ahead_time,
                    "performance_mode": performance_mode,
                    "preload_segments": preload_segments,  # Segment preload hints
                    "segment_length": STREAMING_CONFIG["segment_length"],
                    "optimal_bandwidth": current_video_data.get("bitrate", 3000)  # kbps
                })
                
            position_tracker += video_duration
        
        # Fallback (should not reach here if playlist calculation is correct)
        first_video = valid_videos[0] if valid_videos else {"id": user_video_sequence[0], "duration": 0}
        first_video_id = first_video["id"]
        first_video_data = video_metadata.get(first_video_id, {})
        
        logger.warning(f"Playlist calculation failed, returning first video as fallback: {first_video_id}")
        
        return jsonify({
            "video_id": first_video_id,
            "id": first_video_id,  # For API consistency
            "title": first_video_data.get("name", "Unknown"),
            "name": first_video_data.get("name", "Unknown"),  # For backward compatibility
            "position": 0,
            "duration": first_video.get("duration", 0),
            "elapsed_total": elapsed_seconds,
            "server_time": int(current_time.timestamp() * 1000),
            "stream_start_time": int(start_time.timestamp() * 1000),  # Stream start time in milliseconds
            "sync_interval": 2000,  # Default sync interval
            "hls_enabled": first_video_data.get("hls_url") is not None,
            "hls_url": first_video_data.get("hls_url"),
            "thumbnail_url": first_video_data.get("thumbnail_url"),
            "force_sync": True,  # Flag to enforce client-side synchronization
            "buffer_ahead_time": 15,  # Default buffer ahead time
            "performance_mode": "smooth"  # Default performance mode
        })
    except Exception as e:
        logger.error(f"Error in get_current_video: {e}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 500

@app.route('/api/stream-video/<video_id>', methods=['GET'])
def stream_video(video_id):
    """Legacy endpoint to stream the entire video file directly"""
    if video_id not in video_metadata:
        return jsonify({"error": "Video not found"}), 404
    
    video_path = os.path.join(VIDEOS_DIR, video_metadata[video_id]["path"])
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    # For backward compatibility, we'll still support direct file streaming
    return send_file(video_path)

@app.route('/api/stream-hls/<video_id>/<path:file_path>', methods=['GET', 'OPTIONS'])
def stream_hls_file(video_id, file_path):
    # Add CORS headers to allow the video to be played cross-origin
    if request.method == 'OPTIONS':
        return ('', 204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400'  # 24 hours
        })
    
    # Get the video metadata
    if video_id not in video_metadata:
        return jsonify({"error": "Video not found"}), 404
    
    # Get the path to the HLS files
    streams_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streams")
    video_streams_dir = os.path.join(streams_dir, video_id)
    
    # Determine if this is a segment file (.ts) or playlist (.m3u8)
    is_segment = file_path.endswith('.ts')
    is_playlist = file_path.endswith('.m3u8')
    
    # Full path to the requested file
    full_path = os.path.join(video_streams_dir, file_path)
    
    if not os.path.exists(full_path):
        return jsonify({"error": "Stream file not found"}), 404
    
    # Optimize response headers based on file type
    mimetype = 'application/vnd.apple.mpegurl' if is_playlist else 'video/mp2t'
    
    # Calculate cache time - segments can be cached longer than playlists
    cache_time = STREAMING_CONFIG["cache_ttl"] if is_segment else 5  # 5 seconds for playlists
    
    # Send the file with appropriate headers
    response = send_file(full_path, mimetype=mimetype, conditional=True)
    
    # Add optimization headers
    response.headers['Cache-Control'] = f'public, max-age={cache_time}'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    # Add performance headers
    if is_segment:
        # HTTP/2 Server Push hint for next segment
        segment_number = int(file_path.split('_')[-1].split('.')[0])
        next_segment = f"{file_path.rsplit('_', 1)[0]}_{segment_number + 1}.ts"
        next_segment_path = os.path.join(video_streams_dir, next_segment)
        
        if os.path.exists(next_segment_path):
            # Add Link preload header for the next segment
            response.headers['Link'] = f'</api/stream-hls/{video_id}/{next_segment}>; rel=preload; as=video'
    
    return response

@app.route('/api/thumbnails/<video_id>/<thumbnail_file>', methods=['GET'])
def serve_thumbnail(video_id, thumbnail_file):
    """Serve video thumbnail images with appropriate caching headers"""
    if video_id not in video_metadata:
        return jsonify({"error": "Video not found"}), 404
    
    # Path to the thumbnails directory
    thumbnails_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnails")
    
    # Build the thumbnail path
    thumbnail_path = os.path.join(thumbnails_dir, thumbnail_file)
    
    if not os.path.exists(thumbnail_path):
        return jsonify({"error": "Thumbnail not found"}), 404
    
    # Set caching headers for thumbnails
    response = send_file(thumbnail_path, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'max-age=86400'  # Cache for 1 day
    
    return response

@app.route('/api/list-videos', methods=['GET'])
def list_videos_detailed():
    """Return list of videos with metadata and file existence status"""
    response_videos = {}
    
    for video_id, metadata in video_metadata.items():
        video_path = os.path.join(VIDEOS_DIR, metadata.get("path", ""))
        file_exists = os.path.exists(video_path)
        
        response_videos[video_id] = {
            **metadata,
            "file_exists": file_exists
        }
    
    return jsonify({"videos": response_videos})

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """Legacy endpoint for listing videos"""
    return jsonify({"videos": video_metadata})

@app.route('/api/users', methods=['GET'])
def list_users():
    return jsonify({"users": user_streams})

@app.route('/api/status', methods=['GET'])
def get_status():
    status_info = {
        "global_start_time": global_start_time,
        "running": global_start_time is not None,
        "video_count": len(video_metadata),
        "user_count": len(user_streams)
    }
    
    logger.info(f"Status request: running={status_info['running']}, videos={status_info['video_count']}, users={status_info['user_count']}")
    
    return jsonify(status_info)

@app.route('/api/clear-videos', methods=['POST'])
def clear_videos():
    """Admin endpoint to clear all videos and associated data"""
    global global_start_time
    
    try:
        # First check if there are real videos to clear
        if not video_metadata:
            logger.info("No videos to clear")
            return jsonify({
                'success': True,
                'message': 'No videos to clear'
            })
        
        # Reset the global stream if it's running
        if global_start_time:
            logger.info("Stopping global stream as part of clear videos operation")
            old_time = global_start_time
            global_start_time = None
            
            # Remove global start time file
            start_time_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_start_time.txt")
            if os.path.exists(start_time_path):
                os.unlink(start_time_path)
                logger.info(f"Removed global start time file: {start_time_path}")
        
        # Clear video metadata
        video_count = len(video_metadata)
        video_metadata.clear()
        save_video_data()
        logger.info(f"Cleared metadata for {video_count} videos")
        
        # Clear user streams video references
        users_updated = 0
        for user_id in user_streams:
            if 'videos' in user_streams[user_id]:
                user_streams[user_id]['videos'] = []
                users_updated += 1
        save_user_data()
        logger.info(f"Cleared video lists for {users_updated} users")
        
        # Delete video files
        files_deleted = 0
        for filename in os.listdir(VIDEOS_DIR):
            if filename.endswith('.mp4'):
                file_path = os.path.join(VIDEOS_DIR, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted video file: {file_path}")
                    files_deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        logger.info(f"Deleted {files_deleted} video files")
        
        # Clear thumbnails directory
        thumbnails_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbnails')
        if os.path.exists(thumbnails_dir):
            try:
                shutil.rmtree(thumbnails_dir)
                os.makedirs(thumbnails_dir)
                logger.info(f"Cleared and recreated thumbnails directory")
            except Exception as e:
                logger.error(f"Error clearing thumbnails directory: {e}")
        
        # Clear streams directory
        streams_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streams')
        if os.path.exists(streams_dir):
            try:
                shutil.rmtree(streams_dir)
                os.makedirs(streams_dir)
                logger.info(f"Cleared and recreated streams directory")
            except Exception as e:
                logger.error(f"Error clearing streams directory: {e}")
                
        return jsonify({
            'success': True,
            'message': 'All videos and associated data cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error in clear_videos: {e}")
        return jsonify({
            'success': False,
            'message': f'Error clearing videos: {str(e)}'
        }), 500

@app.route('/api/video/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a specific video by its ID"""
    if video_id not in video_metadata:
        return jsonify({"error": "Video not found"}), 404
    
    # Get the file path
    video_path = os.path.join(VIDEOS_DIR, video_metadata[video_id]["path"])
    
    # Delete the video file if it exists
    if os.path.exists(video_path):
        try:
            os.unlink(video_path)
            logger.info(f"Deleted video file: {video_path}")
        except Exception as e:
            logger.error(f"Error deleting video file {video_path}: {e}")
            return jsonify({"error": f"Error deleting video file: {str(e)}"}), 500
    
    # Delete the thumbnails
    thumbnails_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnails")
    try:
        # Delete all files in thumbnails directory that start with the video_id
        for filename in os.listdir(thumbnails_dir):
            if filename.startswith(f"{video_id}_"):
                file_path = os.path.join(thumbnails_dir, filename)
                os.unlink(file_path)
                logger.info(f"Deleted thumbnail: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting thumbnails for video {video_id}: {e}")
    
    # Delete the stream files
    streams_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streams", video_id)
    if os.path.exists(streams_dir):
        try:
            shutil.rmtree(streams_dir)
            logger.info(f"Deleted stream directory: {streams_dir}")
        except Exception as e:
            logger.error(f"Error deleting stream directory for video {video_id}: {e}")
    
    # Remove the video from metadata
    del video_metadata[video_id]
    save_video_data()
    
    # Remove the video from all user streams
    for user_id in user_streams:
        if video_id in user_streams[user_id]["videos"]:
            user_streams[user_id]["videos"].remove(video_id)
    save_user_data()
    
    return jsonify({"status": "success", "message": "Video deleted successfully"})

@app.route('/api/stop-global-stream', methods=['POST'])
def stop_global_stream():
    """Stop the global stream without deleting videos"""
    global global_start_time
    
    if not global_start_time:
        logger.warning("Attempted to stop stream when it was not running")
        return jsonify({"status": "warning", "message": "Stream was not running"})
    
    # Reset global stream status
    old_time = global_start_time
    global_start_time = None
    
    logger.info(f"Global stream stopped (was running since {old_time})")
    
    start_time_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_start_time.txt")
    if os.path.exists(start_time_file):
        os.unlink(start_time_file)
        logger.info(f"Removed global start time file: {start_time_file}")
    
    return jsonify({"status": "success", "message": "Global stream stopped"})

@app.route('/api/create-stream', methods=['POST'])
def create_stream():
    """Create a new stream with selected videos"""
    data = request.json
    user_id = data.get('user_id')
    stream_name = data.get('name', 'Untitled Stream')
    video_ids = data.get('video_ids', [])
    is_public = data.get('is_public', True)
    
    if not user_id or user_id not in user_streams:
        return jsonify({"error": "Invalid user ID"}), 400
    
    if not video_ids:
        return jsonify({"error": "No videos selected for stream"}), 400
    
    # Validate video IDs
    valid_videos = []
    for video_id in video_ids:
        if video_id in video_metadata:
            valid_videos.append(video_id)
    
    if not valid_videos:
        return jsonify({"error": "No valid videos selected"}), 400
    
    # Create a unique stream ID
    stream_id = str(uuid.uuid4())
    
    # Store stream information
    stream_start_time = datetime.now().isoformat()
    
    active_streams[stream_id] = {
        "id": stream_id,
        "name": stream_name,
        "owner": user_id,
        "video_ids": valid_videos,
        "start_time": stream_start_time,
        "viewers": [user_id],  # Owner is automatically a viewer
        "is_public": is_public,
        "created_at": datetime.now().isoformat()
    }
    
    logger.info(f"Created new stream '{stream_name}' with ID {stream_id} by user {user_id}, with {len(valid_videos)} videos")
    
    # Add this stream to the user's streams
    if "streams" not in user_streams[user_id]:
        user_streams[user_id]["streams"] = []
    
    user_streams[user_id]["streams"].append(stream_id)
    
    # Save updated user data
    save_user_data()
    
    return jsonify({
        "status": "success",
        "stream_id": stream_id,
        "start_time": stream_start_time
    })

@app.route('/api/join-stream/<stream_id>', methods=['POST'])
def join_stream(stream_id):
    """Join an existing stream as a viewer"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id or user_id not in user_streams:
        return jsonify({"error": "Invalid user ID"}), 400
    
    if stream_id not in active_streams:
        return jsonify({"error": "Stream not found"}), 404
    
    # Check if stream is public or user is owner
    stream_info = active_streams[stream_id]
    if not stream_info["is_public"] and user_id != stream_info["owner"]:
        return jsonify({"error": "This stream is private"}), 403
    
    # Add user to viewers if not already there
    if user_id not in stream_info["viewers"]:
        stream_info["viewers"].append(user_id)
        logger.info(f"User {user_id} joined stream {stream_id}, now has {len(stream_info['viewers'])} viewers")
    
    return jsonify({
        "status": "success",
        "stream_id": stream_id,
        "start_time": stream_info["start_time"],
        "name": stream_info["name"]
    })

@app.route('/api/leave-stream/<stream_id>', methods=['POST'])
def leave_stream(stream_id):
    """Leave a stream you're viewing"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id or user_id not in user_streams:
        return jsonify({"error": "Invalid user ID"}), 400
    
    if stream_id not in active_streams:
        return jsonify({"error": "Stream not found"}), 404
    
    # Remove user from viewers
    stream_info = active_streams[stream_id]
    if user_id in stream_info["viewers"]:
        stream_info["viewers"].remove(user_id)
        logger.info(f"User {user_id} left stream {stream_id}, now has {len(stream_info['viewers'])} viewers")
    
    # If stream has no viewers and isn't owned by anyone, clean it up
    if not stream_info["viewers"] and user_id == stream_info["owner"]:
        logger.info(f"Stream {stream_id} has no viewers and owner left, removing")
        del active_streams[stream_id]
    
    return jsonify({
        "status": "success"
    })

@app.route('/api/get-stream-video/<stream_id>/<user_id>', methods=['GET'])
def get_stream_video(stream_id, user_id):
    """Get the current video for a specific stream"""
    if user_id not in user_streams:
        logger.warning(f"Unknown user requested stream video: {user_id}")
        return jsonify({
            "error": "User not found", 
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 404
    
    if stream_id not in active_streams:
        logger.warning(f"Requested non-existent stream: {stream_id}")
        return jsonify({
            "error": "Stream not found", 
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 404
    
    # Verify user is a viewer of this stream
    stream_info = active_streams[stream_id]
    if user_id not in stream_info["viewers"] and not stream_info["is_public"]:
        logger.warning(f"User {user_id} tried to access private stream {stream_id} without being a viewer")
        return jsonify({
            "error": "Not authorized to view this stream", 
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 403
    
    try:
        # Calculate elapsed time since stream started
        start_time = datetime.fromisoformat(stream_info["start_time"])
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        
        logger.info(f"Stream video request: User={user_id}, Stream={stream_id}, Running for {elapsed_seconds:.6f} seconds")
        
        # Get video sequence for this stream
        video_sequence = stream_info["video_ids"]
        
        if not video_sequence:
            return jsonify({
                "error": "No videos in stream",
                "server_time": int(current_time.timestamp() * 1000)
            }), 404
        
        # Calculate total playlist duration and valid videos
        total_duration = 0
        valid_videos = []
        video_start_times = {}
        cumulative_time = 0
        
        for video_id in video_sequence:
            if video_id in video_metadata:
                metadata = video_metadata.get(video_id, {})
                duration = metadata.get("duration", 0)
                if duration > 0:
                    # Record start time of this video in sequence
                    video_start_times[video_id] = cumulative_time
                    
                    total_duration += duration
                    cumulative_time += duration
                    valid_videos.append({
                        "id": video_id,
                        "duration": duration,
                        "title": metadata.get("name", "Unknown"),
                        "hls_enabled": metadata.get("hls_url") is not None,
                        "hls_url": metadata.get("hls_url"),
                        "thumbnail_url": metadata.get("thumbnail_url"),
                        "start_time_in_sequence": video_start_times[video_id]
                    })
            else:
                logger.warning(f"Video {video_id} in stream {stream_id} not found in metadata")
        
        # Handle empty or invalid playlist
        if total_duration <= 0 or not valid_videos:
            logger.warning(f"No valid videos in stream {stream_id}")
            return jsonify({
                "error": "No valid videos in stream",
                "server_time": int(current_time.timestamp() * 1000)
            }), 404
        
        # Calculate position in looping playlist with microsecond precision
        looped_position = elapsed_seconds % total_duration if total_duration > 0 else 0
        
        # Find current video in sequence
        position_tracker = 0
        current_video = None
        position_in_video = 0
        next_video = None
        
        for video in valid_videos:
            video_duration = video["duration"]
            
            if position_tracker + video_duration > looped_position:
                # This is the current video
                current_video = video["id"]
                position_in_video = looped_position - position_tracker
                
                # Determine next video for preloading
                next_idx = (valid_videos.index(video) + 1) % len(valid_videos)
                next_video = valid_videos[next_idx]["id"]
                
                # Calculate time until next transition
                time_until_next = video_duration - position_in_video
                
                # Get current video metadata
                current_video_data = video_metadata.get(current_video, {})
                
                # Calculate precise server times
                precise_server_time = int(current_time.timestamp() * 1000)
                precise_start_time = int(start_time.timestamp() * 1000)
                
                # Select buffer profile based on video duration
                buffer_profile = "medium"
                if video_duration < 60:
                    buffer_profile = "short"
                elif video_duration > 300:
                    buffer_profile = "long"
                
                # Get buffer settings
                buffer_config = STREAMING_CONFIG["buffer_profile"][buffer_profile]
                buffer_ahead_time = buffer_config["ahead"]
                sync_interval = buffer_config["sync_interval"]
                prefetch_count = buffer_config["prefetch"]
                
                # Calculate performance mode
                position_ratio = position_in_video / video_duration if video_duration > 0 else 0
                performance_mode = "smooth"
                
                # Special handling for transition points
                if position_ratio < 0.1 or time_until_next < 10:
                    performance_mode = "aggressive"
                    buffer_ahead_time = max(buffer_ahead_time, 20)
                
                # Calculate keyframe position
                keyframe_position = position_in_video
                
                # Prepare segment preload hints
                preload_segments = []
                
                # Add HLS segment preload hints
                if current_video_data.get("hls_url"):
                    base_segment_path = f"/api/stream-hls/{current_video}/segment"
                    current_segment = int(position_in_video / STREAMING_CONFIG["segment_length"])
                    
                    # Add next segments as preload hints
                    for i in range(current_segment + 1, current_segment + prefetch_count + 1):
                        segment_time = i * STREAMING_CONFIG["segment_length"]
                        if segment_time < video_duration:
                            preload_segments.append({
                                "url": f"{base_segment_path}_{i}.ts",
                                "time_offset": segment_time - position_in_video
                            })
                
                # Add next video preload hints for transitions
                if time_until_next < 10 and next_video:
                    base_segment_path = f"/api/stream-hls/{next_video}/segment"
                    for i in range(0, 3):
                        preload_segments.append({
                            "url": f"{base_segment_path}_{i}.ts",
                            "time_offset": time_until_next + (i * STREAMING_CONFIG["segment_length"])
                        })
                
                logger.info(f"Stream {stream_id} current video: {current_video}, position: {position_in_video:.6f}s")
                
                # Return enhanced response with stream information
                return jsonify({
                    "video_id": current_video,
                    "id": current_video,
                    "title": current_video_data.get("name", "Unknown"),
                    "name": current_video_data.get("name", "Unknown"),
                    "position": position_in_video,
                    "keyframe_position": keyframe_position,
                    "duration": video_duration,
                    "elapsed_total": elapsed_seconds,
                    "next_video": next_video,
                    "time_until_next": time_until_next,
                    "server_time": precise_server_time,
                    "playlist_duration": total_duration,
                    "playlist_position": looped_position,
                    "stream_start_time": precise_start_time,
                    "sync_interval": sync_interval,
                    "hls_enabled": current_video_data.get("hls_url") is not None,
                    "hls_url": current_video_data.get("hls_url"),
                    "thumbnail_url": current_video_data.get("thumbnail_url"),
                    "force_sync": True,
                    "buffer_ahead_time": buffer_ahead_time,
                    "performance_mode": performance_mode,
                    "preload_segments": preload_segments,
                    "segment_length": STREAMING_CONFIG["segment_length"],
                    "optimal_bandwidth": current_video_data.get("bitrate", 3000),
                    "stream_id": stream_id,
                    "stream_name": stream_info["name"],
                    "viewer_count": len(stream_info["viewers"]),
                    "is_owner": user_id == stream_info["owner"]
                })
                
            position_tracker += video_duration
        
        # Fallback - shouldn't reach here
        logger.warning(f"Position calculation failed for stream {stream_id}, falling back to first video")
        
        first_video = valid_videos[0]
        first_video_id = first_video["id"]
        first_video_data = video_metadata.get(first_video_id, {})
        
        return jsonify({
            "video_id": first_video_id,
            "id": first_video_id,
            "title": first_video_data.get("name", "Unknown"),
            "name": first_video_data.get("name", "Unknown"),
            "position": 0,
            "duration": first_video.get("duration", 0),
            "elapsed_total": elapsed_seconds,
            "server_time": int(current_time.timestamp() * 1000),
            "stream_start_time": int(start_time.timestamp() * 1000),
            "sync_interval": 2000,
            "hls_enabled": first_video_data.get("hls_url") is not None,
            "hls_url": first_video_data.get("hls_url"),
            "thumbnail_url": first_video_data.get("thumbnail_url"),
            "force_sync": True,
            "buffer_ahead_time": 15,
            "performance_mode": "smooth",
            "stream_id": stream_id,
            "stream_name": stream_info["name"],
            "viewer_count": len(stream_info["viewers"]),
            "is_owner": user_id == stream_info["owner"]
        })
    except Exception as e:
        logger.error(f"Error in get_stream_video: {e}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "server_time": int(datetime.now().timestamp() * 1000)
        }), 500

@app.route('/api/list-streams', methods=['GET'])
def list_streams():
    """List all public streams and streams the user is part of"""
    user_id = request.args.get('user_id')
    
    result_streams = []
    
    for stream_id, stream_info in active_streams.items():
        # Include stream if it's public or user is owner/viewer
        if stream_info["is_public"] or (user_id and (user_id == stream_info["owner"] or user_id in stream_info["viewers"])):
            # Calculate how long the stream has been running
            start_time = datetime.fromisoformat(stream_info["start_time"])
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            
            # Get thumbnail from first video if available
            thumbnail_url = None
            if stream_info["video_ids"]:
                first_video = stream_info["video_ids"][0]
                if first_video in video_metadata:
                    thumbnail_url = video_metadata[first_video].get("thumbnail_url")
            
            result_streams.append({
                "id": stream_id,
                "name": stream_info["name"],
                "owner": stream_info["owner"],
                "viewer_count": len(stream_info["viewers"]),
                "is_public": stream_info["is_public"],
                "created_at": stream_info["created_at"],
                "thumbnail": thumbnail_url,
                "duration": elapsed_seconds,
                "is_owner": user_id == stream_info["owner"] if user_id else False,
                "is_viewer": user_id in stream_info["viewers"] if user_id else False,
                "video_count": len(stream_info["video_ids"])
            })
    
    # Sort by viewer count (most popular first)
    result_streams.sort(key=lambda x: x["viewer_count"], reverse=True)
    
    return jsonify({
        "streams": result_streams
    })

@app.route('/api/end-stream/<stream_id>', methods=['POST'])
def end_stream(stream_id):
    """End a stream (owner only)"""
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID required"}), 400
    
    if stream_id not in active_streams:
        return jsonify({"error": "Stream not found"}), 404
    
    # Verify user is the stream owner
    stream_info = active_streams[stream_id]
    if user_id != stream_info["owner"]:
        return jsonify({"error": "Only the stream owner can end the stream"}), 403
    
    # Remove the stream
    del active_streams[stream_id]
    
    # Remove from any user's stream list
    for u_id, u_data in user_streams.items():
        if "streams" in u_data and stream_id in u_data["streams"]:
            u_data["streams"].remove(stream_id)
    
    save_user_data()
    
    logger.info(f"Stream {stream_id} ended by owner {user_id}")
    
    return jsonify({
        "status": "success",
        "message": "Stream ended successfully"
    })

if __name__ == '__main__':
    # Load global start time if it exists
    global_start_time_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_start_time.txt")
    
    if os.path.exists(global_start_time_file):
        try:
            with open(global_start_time_file, 'r') as f:
                global_start_time = f.read().strip()
            logger.info(f"Loaded existing global start time: {global_start_time}")
        except Exception as e:
            logger.error(f"Error loading global start time: {e}")
            global_start_time = None
    else:
        logger.info("No global start time file found, stream is not running")
        global_start_time = None
    
    # Report initial stream status
    status = "running" if global_start_time else "stopped"
    logger.info(f"Initial stream status: {status}")
    logger.info(f"Video count: {len(video_metadata)}")
    logger.info(f"User count: {len(user_streams)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 