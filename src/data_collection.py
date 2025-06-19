"""Handles robust data collection from the YouTube API, including replies."""
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pathlib import Path
import time

def get_youtube_comments(api_key: str, video_id: str, output_path: Path = None) -> list:
    """Fetches all top-level comments and their replies from a YouTube video.

    Args:
        api_key: Your Google API key for the YouTube Data API.
        video_id: The ID of the YouTube video.
        output_path: Optional path to save incremental progress.

    Returns:
        A list of all comment and reply texts.
    """
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "YouTube API key not provided or still using placeholder. Please:\n"
            "1. Copy .env.template to .env\n"
            "2. Add your actual YouTube API key to the .env file\n"
            "3. Get your key from: https://console.cloud.google.com/"
        )

    youtube = build('youtube', 'v3', developerKey=api_key)
    all_comments = []
    
    print(f"🎯 Fetching comments for video ID: {video_id}")
    print("📡 Connecting to YouTube API...")
    
    # Check if there's an existing partial file to resume from
    if output_path and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            all_comments = existing_df['Comment'].tolist()
            print(f"📂 Resuming from existing file with {len(all_comments)} comments")
        except:
            print("📂 Starting fresh collection")
    
    try:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        
        page_count = 0
        MAX_PAGES = 1000
        SAVE_INTERVAL = 10  # Save every 10 pages
        
        while request and page_count < MAX_PAGES:
            try:
                response = request.execute()
                page_count += 1
                page_comments = []
                
                print(f"📄 Processing page {page_count}...")
                
                for item in response['items']:
                    # Get top-level comment
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    page_comments.append(comment)
                    
                    # Get replies if they exist
                    if item['snippet']['totalReplyCount'] > 0 and 'replies' in item:
                        for reply_item in item['replies']['comments']:
                            reply_text = reply_item['snippet']['textDisplay']
                            page_comments.append(reply_text)
                
                # Add page comments to total
                all_comments.extend(page_comments)
                
                # INCREMENTAL SAVING - Save every SAVE_INTERVAL pages
                if output_path and (page_count % SAVE_INTERVAL == 0):
                    save_comments_to_csv(all_comments, output_path)
                    print(f"💾 Incremental save: {len(all_comments)} comments saved to backup")
                
                # Get next page
                request = youtube.commentThreads().list_next(request, response)
                
                # Rate limiting to be respectful to API
                time.sleep(0.1)
                
                # Progress update every 10 pages
                if page_count % 10 == 0:
                    print(f"💬 Collected {len(all_comments)} comments so far...")
                    
            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted by user! Saving {len(all_comments)} comments collected so far...")
                if output_path:
                    save_comments_to_csv(all_comments, output_path)
                    print(f"✅ Saved progress to {output_path}")
                return all_comments
                
            except HttpError as e:
                print(f"❌ API error on page {page_count}: {e}")
                # Save progress before potentially failing
                if output_path:
                    save_comments_to_csv(all_comments, output_path)
                    print(f"💾 Saved progress before error: {len(all_comments)} comments")
                break

    except HttpError as e:
        error_details = e.content.decode() if hasattr(e, 'content') else str(e)
        print(f"❌ YouTube API error {e.resp.status}: {error_details}")
        
        if e.resp.status == 403:
            print("🔑 This might be an API key issue. Check:")
            print("   - API key is correct in .env file")
            print("   - YouTube Data API v3 is enabled in Google Cloud Console")
            print("   - You haven't exceeded your daily quota")
        elif e.resp.status == 404:
            print(f"📹 Video not found: {video_id}")
            print("   - Check if the video ID is correct")
            print("   - Check if comments are enabled for this video")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        # Save whatever we have
        if output_path and all_comments:
            save_comments_to_csv(all_comments, output_path)
            print(f"💾 Emergency save: {len(all_comments)} comments saved")

    print(f"✅ Successfully fetched {len(all_comments)} total comments and replies!")
    return all_comments

def save_comments_to_csv(comments: list, file_path: Path):
    """Saves a list of comments to a CSV file.

    Args:
        comments: A list of comment strings.
        file_path: The path to save the CSV file.
    """
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame and save (using 'Comment' to match your existing data)
    df = pd.DataFrame(comments, columns=['Comment'])
    df.to_csv(file_path, index=False)
    print(f"💾 Successfully saved {len(comments)} comments to {file_path}")