import asyncio
import json
import os
import time
import uuid
from collections import deque
from urllib.parse import urlencode
from dotenv import load_dotenv
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

load_dotenv()
logger = configure_logger(__name__)


class AssemblyTranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='best', stream=True, language="en", 
                 sampling_rate="16000", encoding="linear16", output_queue=None, 
                 end_of_turn_confidence_threshold=0.7, format_turns=True, **kwargs):
        super().__init__(input_queue)
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.model = model
        self.sampling_rate = int(sampling_rate)
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('ASSEMBLYAI_API_KEY'))
        self.assembly_host = "streaming.assemblyai.com"
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        
        
        self.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        self.format_turns = format_turns
        self.min_end_of_turn_silence_when_confident = 160  # ms
        self.max_turn_silence = 2400  # ms
        
       
        self.websocket_connection = None
        self.connection_authenticated = False
        self.sender_task = None
        self.heartbeat_task = None
        
        
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        self.num_frames = 0
        self.connection_start_time = None
        self.audio_frame_duration = 0.0
        self.audio_submitted = False
        self.audio_submission_time = None
        
       
        self.current_turn_transcript = ""
        self.is_transcript_sent_for_processing = False
        
        
        self.request_times = deque()
        self.max_requests_per_window = 20000
        self.rate_limit_window = 300  # 5 minutes in seconds
        
        
        self.session_start_time = None
        self.max_session_duration = 10800  # 3 hours in seconds
        
        
        self._configure_audio_settings()

    def _configure_audio_settings(self):
        """Configure audio settings based on telephony provider (following Deepgram pattern)"""
        if self.provider in ('twilio', 'exotel', 'plivo'):
            # Telephony providers
            if self.provider == "twilio":
                self.encoding = 'mulaw'
                self.assembly_encoding = 'pcm_mulaw'
            else:
                self.encoding = 'linear16'
                self.assembly_encoding = 'pcm_s16le'
            
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2  # 200ms chunks for telephony
            
        elif self.provider == "web_based_call":
            self.encoding = "linear16"
            self.assembly_encoding = 'pcm_s16le'
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256  # 256ms chunks for web
            
        elif self.provider == "playground":
            self.encoding = "linear16"
            self.assembly_encoding = 'pcm_s16le'
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0  # No streaming from playground
            
        else:
            # Default settings
            self.encoding = "linear16"
            self.assembly_encoding = 'pcm_s16le'
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.5  # 500ms default

    def _check_rate_limits(self):
        """Check if we can make a request without exceeding rate limits"""
        current_time = time.time()
        
        
        while self.request_times and self.request_times[0] < (current_time - self.rate_limit_window):
            self.request_times.popleft()
        
        
        if len(self.request_times) >= self.max_requests_per_window:
            logger.warning("AssemblyAI rate limit approaching, delaying request")
            return False
        
        
        self.request_times.append(current_time)
        return True

    def _should_reconnect_session(self):
        """Check if we need to reconnect due to session duration limit"""
        if not self.session_start_time:
            return False
        
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        
        if session_duration > (self.max_session_duration - 300):
            logger.info("Approaching AssemblyAI session duration limit, will reconnect")
            return True
        
        return False

    def get_assembly_ws_url(self):
        """Build AssemblyAI WebSocket URL with proper parameters"""
        assembly_params = {
            'sample_rate': str(self.sampling_rate),
            'encoding': self.assembly_encoding,
            'format_turns': 'true' if self.format_turns else 'false',
            'end_of_turn_confidence_threshold': str(self.end_of_turn_confidence_threshold),
            'min_end_of_turn_silence_when_confident': str(self.min_end_of_turn_silence_when_confident),
            'max_turn_silence': str(self.max_turn_silence)
        }
        
        websocket_api = f'wss://{self.assembly_host}/v3/ws?'
        websocket_url = websocket_api + urlencode(assembly_params)
        logger.info(f"AssemblyAI WebSocket URL: {websocket_url}")
        return websocket_url

    async def assembly_connect(self):
        """Establish WebSocket connection to AssemblyAI with proper error handling"""
        try:
           
            if not self._check_rate_limits():
                await asyncio.sleep(1)  # Brief delay before retry
                if not self._check_rate_limits():
                    raise ConnectionError("AssemblyAI rate limit exceeded")
            
            websocket_url = self.get_assembly_ws_url()
            additional_headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            logger.info(f"Attempting to connect to AssemblyAI WebSocket: {websocket_url}")
            
            assembly_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, extra_headers=additional_headers),
                timeout=10.0  # 10 second timeout
            )
            
            self.websocket_connection = assembly_ws
            self.connection_authenticated = True
            self.session_start_time = time.time()
            logger.info("Successfully connected to AssemblyAI WebSocket")
            
            return assembly_ws
            
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to AssemblyAI WebSocket")
            raise ConnectionError("Timeout while connecting to AssemblyAI WebSocket")
        except InvalidHandshake as e:
            logger.error(f"Invalid handshake during AssemblyAI WebSocket connection: {e}")
            raise ConnectionError(f"Invalid handshake during AssemblyAI WebSocket connection: {e}")
        except ConnectionClosedError as e:
            logger.error(f"AssemblyAI WebSocket connection closed unexpectedly: {e}")
            raise ConnectionError(f"AssemblyAI WebSocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to AssemblyAI WebSocket: {e}")
            raise ConnectionError(f"Unexpected error connecting to AssemblyAI WebSocket: {e}")

    async def sender_stream(self, ws: ClientConnection):
        """Send audio data to AssemblyAI WebSocket (following existing pattern)"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                
                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                
                
                if self._should_reconnect_session():
                    logger.info("Reconnecting AssemblyAI session due to duration limit")
                    await self._close_connection(ws, {"type": "Terminate"})
                    break
                
                self.num_frames += 1

                self.audio_cursor = self.num_frames * self.audio_frame_duration
                
                try:
                   
                    audio_data = ws_data_packet.get('data')
                    await ws.send(audio_data)
                except ConnectionClosedError as e:
                    logger.error(f"Connection closed while sending data to AssemblyAI: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending data to AssemblyAI WebSocket: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("AssemblyAI sender stream task cancelled")
            raise
        except Exception as e:
            logger.error(f'Error in AssemblyAI sender_stream: {str(e)}')
            raise

    async def receiver(self, ws: ClientConnection):
        """Process messages from AssemblyAI WebSocket"""
        async for msg in ws:
            try:
                msg_data = json.loads(msg)
                
               
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                if msg_data["type"] == "Begin":
                    logger.info(f"AssemblyAI session started: {msg_data.get('id')}")
                    
                    continue

                elif msg_data["type"] == "Turn":
                    transcript = msg_data.get("transcript", "").strip()
                    end_of_turn = msg_data.get("end_of_turn", False)
                    turn_is_formatted = msg_data.get("turn_is_formatted", False)
                    
                    if transcript:
                       
                        if not end_of_turn:
                            data = {
                                "type": "interim_transcript_received",
                                "content": transcript
                            }
                            yield create_ws_data_packet(data, self.meta_info)
                        
                       
                        self.current_turn_transcript = transcript
                        
                       
                        if end_of_turn and not self.is_transcript_sent_for_processing:
                            logger.info(f"AssemblyAI end of turn detected: {transcript}")
                            data = {
                                "type": "transcript",
                                "content": transcript
                            }
                            self.is_transcript_sent_for_processing = True
                            yield create_ws_data_packet(data, self.meta_info)
                            
                            
                            self.current_turn_transcript = ""
                            self.is_transcript_sent_for_processing = False

                elif msg_data["type"] == "Termination":
                    logger.info(f"AssemblyAI session terminated: {msg_data}")
                   
                    if "audio_duration_seconds" in msg_data:
                        self.meta_info["transcriber_duration"] = msg_data["audio_duration_seconds"]
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AssemblyAI message: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing AssemblyAI message: {e}")
                continue

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end of stream signal and close connection if needed"""
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            await self._close_connection(ws, {"type": "Terminate"})
            return True
        return False

    async def _close_connection(self, ws, data):
        """Close AssemblyAI connection gracefully"""
        try:
            await ws.send(json.dumps(data))
            logger.info("Sent termination message to AssemblyAI")
        except Exception as e:
            logger.error(f"Error while closing AssemblyAI connection: {e}")

    async def push_to_transcriber_queue(self, data_packet):
        """Push processed transcript to output queue"""
        await self.transcriber_output_queue.put(data_packet)

    async def toggle_connection(self):
        """Clean up connection and cancel tasks"""
        self.connection_on = False
        
       
        if self.sender_task is not None:
            self.sender_task.cancel()
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        
        
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("AssemblyAI WebSocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing AssemblyAI WebSocket connection: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False
                self.session_start_time = None

    def get_meta_info(self):
        """Get current meta information"""
        return self.meta_info

    async def run(self):
        """Start the transcription task"""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting AssemblyAI transcription task: {e}")

    async def transcribe(self):
        """Main transcription method (following existing pattern)"""
        assembly_ws = None
        try:
            start_time = time.perf_counter()
            
            try:
                assembly_ws = await self.assembly_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish AssemblyAI connection: {e}")
                await self.toggle_connection()
                return
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            if self.stream:
               
                self.sender_task = asyncio.create_task(self.sender_stream(assembly_ws))
                
                try:
                    
                    async for message in self.receiver(assembly_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing the AssemblyAI connection")
                            await self._close_connection(assembly_ws, {"type": "Terminate"})
                            break
                except ConnectionClosedError as e:
                    logger.error(f"AssemblyAI WebSocket connection closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during AssemblyAI streaming: {e}")
                    raise
            else:
                logger.error("AssemblyAI only supports streaming mode")
                raise ValueError("AssemblyAI requires stream=True")

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in AssemblyAI transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in AssemblyAI transcribe: {e}")
            await self.toggle_connection()
        finally:
            
            if assembly_ws is not None:
                try:
                    await assembly_ws.close()
                    logger.info("AssemblyAI WebSocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing AssemblyAI WebSocket in finally block: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
                    self.session_start_time = None
            
            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            
            
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )