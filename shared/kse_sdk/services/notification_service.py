"""
KSE Memory SDK Notification Service
"""

from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
import time
import logging
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from dataclasses import dataclass, field
from ..core.interfaces import NotificationInterface

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CALLBACK = "callback"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationTemplate:
    """Notification template definition."""
    id: str
    name: str
    subject_template: str
    body_template: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """Individual notification instance."""
    id: str
    template_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    recipient: str
    subject: str
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    delivered_at: Optional[float] = None
    failed_at: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class NotificationService(NotificationInterface):
    """
    Notification service for KSE Memory operations including email,
    webhooks, and custom notification channels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.templates: Dict[str, NotificationTemplate] = {}
        self.notifications: Dict[str, Notification] = {}
        self.subscribers: Dict[str, List[str]] = {}  # event -> recipients
        self.callbacks: Dict[str, Callable] = {}
        
        # Configuration - handle both dict and dataclass config
        if hasattr(config, 'get'):
            # Dictionary-style config
            self.smtp_config = config.get('smtp', {})
            self.default_sender = config.get('default_sender', 'noreply@liftos.ai')
            self.webhook_config = config.get('webhooks', {})
        else:
            # Dataclass-style config
            self.smtp_config = getattr(config, 'smtp', {})
            self.default_sender = getattr(config, 'default_sender', 'noreply@liftos.ai')
            self.webhook_config = getattr(config, 'webhooks', {})
        
        # Queue for async processing
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """Initialize notification service."""
        try:
            # Load default templates
            await self._load_default_templates()
            
            # Start notification processing
            self.processing_task = asyncio.create_task(self._process_notifications())
            
            logger.info("Notification service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize notification service: {e}")
            return False
    
    async def create_template(self, template_id: str, name: str,
                            subject_template: str, body_template: str,
                            channel: str, priority: str = "normal",
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a notification template.
        
        Args:
            template_id: Unique template identifier
            name: Template name
            subject_template: Subject template with placeholders
            body_template: Body template with placeholders
            channel: Notification channel
            priority: Priority level
            metadata: Additional metadata
            
        Returns:
            True if created successfully
        """
        try:
            template = NotificationTemplate(
                id=template_id,
                name=name,
                subject_template=subject_template,
                body_template=body_template,
                channel=NotificationChannel(channel),
                priority=NotificationPriority(priority),
                metadata=metadata or {}
            )
            
            self.templates[template_id] = template
            
            logger.info(f"Created notification template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return False
    
    async def send_notification(self, template_id: str, recipient: str,
                              variables: Optional[Dict[str, Any]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Send a notification using a template.
        
        Args:
            template_id: Template to use
            recipient: Notification recipient
            variables: Variables for template substitution
            metadata: Additional metadata
            
        Returns:
            Notification ID if queued successfully
        """
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template not found: {template_id}")
            
            template = self.templates[template_id]
            variables = variables or {}
            
            # Render template
            subject = self._render_template(template.subject_template, variables)
            body = self._render_template(template.body_template, variables)
            
            # Create notification
            notification_id = f"notif_{int(time.time() * 1000)}"
            notification = Notification(
                id=notification_id,
                template_id=template_id,
                channel=template.channel,
                priority=template.priority,
                recipient=recipient,
                subject=subject,
                body=body,
                metadata=metadata or {}
            )
            
            self.notifications[notification_id] = notification
            
            # Queue for processing
            await self.notification_queue.put(notification)
            
            logger.info(f"Queued notification: {notification_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return None
    
    async def send_direct_notification(self, channel: str, recipient: str,
                                     subject: str, body: str,
                                     priority: str = "normal",
                                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Send a direct notification without using a template.
        
        Args:
            channel: Notification channel
            recipient: Notification recipient
            subject: Notification subject
            body: Notification body
            priority: Priority level
            metadata: Additional metadata
            
        Returns:
            Notification ID if queued successfully
        """
        try:
            notification_id = f"notif_{int(time.time() * 1000)}"
            notification = Notification(
                id=notification_id,
                template_id="direct",
                channel=NotificationChannel(channel),
                priority=NotificationPriority(priority),
                recipient=recipient,
                subject=subject,
                body=body,
                metadata=metadata or {}
            )
            
            self.notifications[notification_id] = notification
            
            # Queue for processing
            await self.notification_queue.put(notification)
            
            logger.info(f"Queued direct notification: {notification_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Failed to send direct notification: {e}")
            return None
    
    async def subscribe_to_event(self, event_type: str, recipient: str) -> bool:
        """
        Subscribe a recipient to an event type.
        
        Args:
            event_type: Event type to subscribe to
            recipient: Recipient to subscribe
            
        Returns:
            True if subscribed successfully
        """
        try:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            if recipient not in self.subscribers[event_type]:
                self.subscribers[event_type].append(recipient)
            
            logger.info(f"Subscribed {recipient} to {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to event: {e}")
            return False
    
    async def unsubscribe_from_event(self, event_type: str, recipient: str) -> bool:
        """
        Unsubscribe a recipient from an event type.
        
        Args:
            event_type: Event type to unsubscribe from
            recipient: Recipient to unsubscribe
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            if event_type in self.subscribers and recipient in self.subscribers[event_type]:
                self.subscribers[event_type].remove(recipient)
            
            logger.info(f"Unsubscribed {recipient} from {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from event: {e}")
            return False
    
    async def notify_event(self, event_type: str, template_id: str,
                          variables: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Notify all subscribers of an event.
        
        Args:
            event_type: Event type
            template_id: Template to use
            variables: Variables for template substitution
            metadata: Additional metadata
            
        Returns:
            List of notification IDs
        """
        try:
            if event_type not in self.subscribers:
                return []
            
            notification_ids = []
            
            for recipient in self.subscribers[event_type]:
                notification_id = await self.send_notification(
                    template_id=template_id,
                    recipient=recipient,
                    variables=variables,
                    metadata=metadata
                )
                
                if notification_id:
                    notification_ids.append(notification_id)
            
            logger.info(f"Notified {len(notification_ids)} subscribers of {event_type}")
            return notification_ids
            
        except Exception as e:
            logger.error(f"Failed to notify event: {e}")
            return []
    
    async def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """
        Get notification status.
        
        Args:
            notification_id: Notification ID
            
        Returns:
            Notification status information
        """
        try:
            if notification_id not in self.notifications:
                return None
            
            notification = self.notifications[notification_id]
            
            status = "pending"
            if notification.failed_at:
                status = "failed"
            elif notification.delivered_at:
                status = "delivered"
            elif notification.sent_at:
                status = "sent"
            
            return {
                'id': notification.id,
                'template_id': notification.template_id,
                'channel': notification.channel.value,
                'priority': notification.priority.value,
                'recipient': notification.recipient,
                'status': status,
                'created_at': notification.created_at,
                'sent_at': notification.sent_at,
                'delivered_at': notification.delivered_at,
                'failed_at': notification.failed_at,
                'error': notification.error,
                'retry_count': notification.retry_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get notification status: {e}")
            return None
    
    async def register_callback(self, name: str, callback: Callable) -> bool:
        """
        Register a callback function for notifications.
        
        Args:
            name: Callback name
            callback: Callback function
            
        Returns:
            True if registered successfully
        """
        try:
            self.callbacks[name] = callback
            logger.info(f"Registered notification callback: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register callback: {e}")
            return False
    
    async def _process_notifications(self) -> None:
        """Process notifications from the queue."""
        while True:
            try:
                # Get notification from queue
                notification = await self.notification_queue.get()
                
                # Process based on channel
                success = False
                
                if notification.channel == NotificationChannel.EMAIL:
                    success = await self._send_email(notification)
                elif notification.channel == NotificationChannel.WEBHOOK:
                    success = await self._send_webhook(notification)
                elif notification.channel == NotificationChannel.LOG:
                    success = await self._send_log(notification)
                elif notification.channel == NotificationChannel.CALLBACK:
                    success = await self._send_callback(notification)
                
                # Update notification status
                if success:
                    notification.sent_at = time.time()
                    notification.delivered_at = time.time()
                else:
                    notification.retry_count += 1
                    
                    if notification.retry_count <= notification.max_retries:
                        # Retry after delay
                        await asyncio.sleep(2 ** notification.retry_count)
                        await self.notification_queue.put(notification)
                    else:
                        notification.failed_at = time.time()
                
                # Mark task as done
                self.notification_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    async def _send_email(self, notification: Notification) -> bool:
        """Send email notification."""
        try:
            if not self.smtp_config:
                logger.warning("SMTP not configured, skipping email")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('sender', self.default_sender)
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            # Add body
            msg.attach(MIMEText(notification.body, 'plain'))
            
            # Send email
            with smtplib.SMTP(
                self.smtp_config['host'], 
                self.smtp_config.get('port', 587)
            ) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls()
                
                if 'username' in self.smtp_config:
                    server.login(
                        self.smtp_config['username'],
                        self.smtp_config['password']
                    )
                
                server.send_message(msg)
            
            logger.info(f"Sent email notification to {notification.recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            notification.error = str(e)
            return False
    
    async def _send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification."""
        try:
            import aiohttp
            
            webhook_url = notification.recipient
            payload = {
                'subject': notification.subject,
                'body': notification.body,
                'metadata': notification.metadata,
                'timestamp': notification.created_at
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status < 400:
                        logger.info(f"Sent webhook notification to {webhook_url}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        notification.error = f"HTTP {response.status}"
                        return False
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            notification.error = str(e)
            return False
    
    async def _send_log(self, notification: Notification) -> bool:
        """Send log notification."""
        try:
            log_level = notification.priority.value.upper()
            message = f"[{notification.subject}] {notification.body}"
            
            if log_level == "CRITICAL":
                logger.critical(message)
            elif log_level == "HIGH":
                logger.error(message)
            elif log_level == "NORMAL":
                logger.info(message)
            else:
                logger.debug(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send log notification: {e}")
            notification.error = str(e)
            return False
    
    async def _send_callback(self, notification: Notification) -> bool:
        """Send callback notification."""
        try:
            callback_name = notification.recipient
            
            if callback_name not in self.callbacks:
                logger.error(f"Callback not found: {callback_name}")
                notification.error = f"Callback not found: {callback_name}"
                return False
            
            callback = self.callbacks[callback_name]
            
            # Call the callback
            if asyncio.iscoroutinefunction(callback):
                await callback(notification)
            else:
                callback(notification)
            
            logger.info(f"Called notification callback: {callback_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to call callback: {e}")
            notification.error = str(e)
            return False
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables."""
        try:
            # Simple template rendering using string formatting
            return template.format(**variables)
        except Exception as e:
            logger.error(f"Failed to render template: {e}")
            return template
    
    async def _load_default_templates(self) -> None:
        """Load default notification templates."""
        
        # System alert template
        await self.create_template(
            template_id="system_alert",
            name="System Alert",
            subject_template="LiftOS Alert: {alert_type}",
            body_template="Alert: {message}\nTime: {timestamp}\nSeverity: {severity}",
            channel="email",
            priority="high"
        )
        
        # Workflow completion template
        await self.create_template(
            template_id="workflow_complete",
            name="Workflow Completion",
            subject_template="Workflow Completed: {workflow_name}",
            body_template="Workflow '{workflow_name}' has completed successfully.\nDuration: {duration}\nResults: {results}",
            channel="email",
            priority="normal"
        )
        
        # Error notification template
        await self.create_template(
            template_id="error_notification",
            name="Error Notification",
            subject_template="LiftOS Error: {error_type}",
            body_template="An error occurred: {error_message}\nService: {service}\nTime: {timestamp}",
            channel="log",
            priority="high"
        )
    
    async def subscribe(self, user_id: str, event_type: str, channel: str = "email") -> bool:
        """
        Subscribe user to notifications.
        
        Args:
            user_id: User ID
            event_type: Event type to subscribe to
            channel: Notification channel
            
        Returns:
            True if subscribed successfully
        """
        try:
            # Use the existing subscribe_to_event method
            return await self.subscribe_to_event(event_type, user_id)
        except Exception as e:
            logger.error(f"Failed to subscribe user {user_id} to {event_type}: {e}")
            return False
    
    async def unsubscribe(self, user_id: str, event_type: str, channel: str = "email") -> bool:
        """
        Unsubscribe user from notifications.
        
        Args:
            user_id: User ID
            event_type: Event type to unsubscribe from
            channel: Notification channel
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            # Use the existing unsubscribe_from_event method
            return await self.unsubscribe_from_event(event_type, user_id)
        except Exception as e:
            logger.error(f"Failed to unsubscribe user {user_id} from {event_type}: {e}")
            return False
    
    async def get_notification_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get notification history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of notifications to return
            
        Returns:
            List of notification history entries
        """
        try:
            history = []
            
            # Get notifications for the user
            user_notifications = [
                notification for notification in self.notifications.values()
                if notification.recipient == user_id
            ]
            
            # Sort by creation time (newest first)
            user_notifications.sort(key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            user_notifications = user_notifications[:limit]
            
            # Convert to history format
            for notification in user_notifications:
                status = "pending"
                if notification.failed_at:
                    status = "failed"
                elif notification.delivered_at:
                    status = "delivered"
                elif notification.sent_at:
                    status = "sent"
                
                history.append({
                    'id': notification.id,
                    'template_id': notification.template_id,
                    'subject': notification.subject,
                    'body': notification.body,
                    'channel': notification.channel.value,
                    'priority': notification.priority.value,
                    'status': status,
                    'created_at': notification.created_at,
                    'sent_at': notification.sent_at,
                    'delivered_at': notification.delivered_at,
                    'failed_at': notification.failed_at,
                    'error': notification.error,
                    'retry_count': notification.retry_count,
                    'metadata': notification.metadata
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get notification history for user {user_id}: {e}")
            return []