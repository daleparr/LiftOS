"""
Enhanced JWT Manager for LiftOS
Provides refresh token rotation, device fingerprinting, and enterprise security features
"""

import jwt
import os
import hashlib
import secrets
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import logging

from ..database.security_models import EnhancedUserSession, RevokedToken, SecurityConfiguration
from .audit_logger import SecurityAuditLogger, SecurityEventType

logger = logging.getLogger(__name__)

class DeviceFingerprint:
    """Device fingerprinting for enhanced security"""
    
    @staticmethod
    def generate_fingerprint(
        user_agent: str,
        ip_address: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a device fingerprint from request data"""
        fingerprint_data = {
            'user_agent': user_agent,
            'ip_address': ip_address,
            **(additional_data or {})
        }
        
        # Create a hash of the fingerprint data
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()
    
    @staticmethod
    def is_suspicious_device(
        current_fingerprint: str,
        known_fingerprints: List[str],
        threshold: float = 0.8
    ) -> bool:
        """Check if device fingerprint indicates suspicious activity"""
        if not known_fingerprints:
            return False
        
        # Simple check - in production, use more sophisticated similarity algorithms
        return current_fingerprint not in known_fingerprints

class EnhancedJWTManager:
    """
    Enterprise JWT manager with refresh tokens, device fingerprinting, and security features
    """
    
    def __init__(self):
        self.access_secret = os.getenv("JWT_ACCESS_SECRET", "access-secret-change-in-production")
        self.refresh_secret = os.getenv("JWT_REFRESH_SECRET", "refresh-secret-change-in-production")
        self.algorithm = "RS256"  # Use RSA for enhanced security
        self.access_token_expire_minutes = 15  # Short-lived access tokens
        self.refresh_token_expire_days = 30    # Longer-lived refresh tokens
        self.audit_logger = SecurityAuditLogger()
        
        # Load RSA keys (in production, load from secure storage)
        self._load_rsa_keys()
    
    def _load_rsa_keys(self):
        """Load RSA private and public keys"""
        # In production, load these from secure key management
        private_key_path = os.getenv("JWT_PRIVATE_KEY_PATH")
        public_key_path = os.getenv("JWT_PUBLIC_KEY_PATH")
        
        if private_key_path and public_key_path:
            try:
                with open(private_key_path, 'r') as f:
                    self.private_key = f.read()
                with open(public_key_path, 'r') as f:
                    self.public_key = f.read()
                logger.info("Loaded RSA keys from files")
                return
            except Exception as e:
                logger.warning(f"Failed to load RSA keys from files: {e}")
        
        # Fallback to HMAC for development
        self.algorithm = "HS256"
        self.private_key = self.access_secret
        self.public_key = self.access_secret
        logger.warning("Using HMAC keys - generate RSA keys for production!")
    
    def _create_token(
        self,
        payload: Dict[str, Any],
        secret: str,
        expire_delta: timedelta
    ) -> str:
        """Create a JWT token with the given payload and expiration"""
        now = datetime.now(timezone.utc)
        payload.update({
            "iat": now,
            "exp": now + expire_delta,
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        })
        
        if self.algorithm.startswith("RS"):
            return jwt.encode(payload, self.private_key, algorithm=self.algorithm)
        else:
            return jwt.encode(payload, secret, algorithm=self.algorithm)
    
    def _verify_token(self, token: str, secret: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            if self.algorithm.startswith("RS"):
                payload = jwt.decode(token, self.public_key, algorithms=[self.algorithm])
            else:
                payload = jwt.decode(token, secret, algorithms=[self.algorithm])
            
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")
    
    async def create_token_pair(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        email: str,
        roles: List[str],
        permissions: List[str],
        device_fingerprint: str,
        ip_address: str,
        user_agent: str,
        location: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Create access and refresh token pair with session tracking
        Returns: (access_token, refresh_token, session_info)
        """
        try:
            # Create access token payload
            access_payload = {
                "sub": user_id,
                "org_id": org_id,
                "email": email,
                "roles": roles,
                "permissions": permissions,
                "type": "access",
                "device_fp": device_fingerprint[:16]  # Truncated fingerprint for token
            }
            
            # Create refresh token payload
            refresh_payload = {
                "sub": user_id,
                "org_id": org_id,
                "type": "refresh",
                "device_fp": device_fingerprint[:16]
            }
            
            # Generate tokens
            access_token = self._create_token(
                access_payload,
                self.access_secret,
                timedelta(minutes=self.access_token_expire_minutes)
            )
            
            refresh_token = self._create_token(
                refresh_payload,
                self.refresh_secret,
                timedelta(days=self.refresh_token_expire_days)
            )
            
            # Extract JTIs for session tracking
            access_jti = jwt.decode(access_token, options={"verify_signature": False})["jti"]
            refresh_jti = jwt.decode(refresh_token, options={"verify_signature": False})["jti"]
            
            # Check for existing sessions and enforce limits
            await self._enforce_session_limits(session, user_id, org_id)
            
            # Create session record
            user_session = EnhancedUserSession(
                user_id=user_id,
                access_token_jti=access_jti,
                refresh_token_jti=refresh_jti,
                device_fingerprint=device_fingerprint,
                ip_address=ip_address,
                user_agent=user_agent,
                location=location,
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days),
                is_active=True
            )
            
            session.add(user_session)
            await session.flush()
            
            # Log successful token creation
            await self.audit_logger.log_security_event(
                session=session,
                event_type=SecurityEventType.LOGIN_SUCCESS,
                user_id=user_id,
                org_id=org_id,
                action="token_created",
                ip_address=ip_address,
                user_agent=user_agent,
                success=True,
                details={
                    "session_id": str(user_session.id),
                    "device_fingerprint": device_fingerprint[:16]
                }
            )
            
            await session.commit()
            
            session_info = {
                "session_id": str(user_session.id),
                "expires_at": user_session.expires_at.isoformat(),
                "device_fingerprint": device_fingerprint
            }
            
            return access_token, refresh_token, session_info
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to create token pair: {e}")
            raise
    
    async def refresh_access_token(
        self,
        session: AsyncSession,
        refresh_token: str,
        device_fingerprint: str,
        ip_address: str,
        user_agent: str
    ) -> Tuple[str, str]:
        """
        Refresh access token using refresh token
        Returns: (new_access_token, new_refresh_token)
        """
        try:
            # Verify refresh token
            refresh_payload = self._verify_token(refresh_token, self.refresh_secret)
            
            if refresh_payload.get("type") != "refresh":
                raise ValueError("Invalid token type")
            
            refresh_jti = refresh_payload["jti"]
            user_id = refresh_payload["sub"]
            org_id = refresh_payload["org_id"]
            
            # Check if token is revoked
            revoked_token = await session.execute(
                select(RevokedToken).where(RevokedToken.token_jti == refresh_jti)
            )
            if revoked_token.scalar_one_or_none():
                raise ValueError("Token has been revoked")
            
            # Find active session
            user_session_result = await session.execute(
                select(EnhancedUserSession).where(
                    EnhancedUserSession.refresh_token_jti == refresh_jti,
                    EnhancedUserSession.is_active == True
                )
            )
            user_session = user_session_result.scalar_one_or_none()
            
            if not user_session:
                raise ValueError("Session not found or inactive")
            
            # Verify device fingerprint for security
            if user_session.device_fingerprint != device_fingerprint:
                # Log suspicious activity
                await self.audit_logger.log_suspicious_activity(
                    session=session,
                    user_id=user_id,
                    ip_address=ip_address,
                    activity_type="device_fingerprint_mismatch",
                    details={
                        "expected_fingerprint": user_session.device_fingerprint[:16],
                        "received_fingerprint": device_fingerprint[:16]
                    }
                )
                raise ValueError("Device fingerprint mismatch")
            
            # Get user info for new tokens (you'd typically fetch from user table)
            # For now, we'll use basic info
            email = f"user_{user_id}@example.com"  # Replace with actual user lookup
            roles = ["user"]  # Replace with actual role lookup
            permissions = ["read"]  # Replace with actual permission lookup
            
            # Revoke old tokens
            await self._revoke_session_tokens(session, user_session, "token_refresh")
            
            # Create new token pair
            new_access_token, new_refresh_token, session_info = await self.create_token_pair(
                session=session,
                user_id=user_id,
                org_id=org_id,
                email=email,
                roles=roles,
                permissions=permissions,
                device_fingerprint=device_fingerprint,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Update session activity
            await session.execute(
                update(EnhancedUserSession)
                .where(EnhancedUserSession.id == user_session.id)
                .values(
                    is_active=False,  # Deactivate old session
                    last_activity=datetime.now(timezone.utc)
                )
            )
            
            logger.info(f"Token refreshed for user {user_id}")
            return new_access_token, new_refresh_token
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to refresh token: {e}")
            raise
    
    async def verify_access_token(
        self,
        session: AsyncSession,
        access_token: str
    ) -> Dict[str, Any]:
        """Verify access token and return payload"""
        try:
            payload = self._verify_token(access_token, self.access_secret)
            
            if payload.get("type") != "access":
                raise ValueError("Invalid token type")
            
            access_jti = payload["jti"]
            
            # Check if token is revoked
            revoked_token = await session.execute(
                select(RevokedToken).where(RevokedToken.token_jti == access_jti)
            )
            if revoked_token.scalar_one_or_none():
                raise ValueError("Token has been revoked")
            
            # Update session activity
            await session.execute(
                update(EnhancedUserSession)
                .where(
                    EnhancedUserSession.access_token_jti == access_jti,
                    EnhancedUserSession.is_active == True
                )
                .values(last_activity=datetime.now(timezone.utc))
            )
            
            await session.commit()
            return payload
            
        except Exception as e:
            logger.error(f"Failed to verify access token: {e}")
            raise
    
    async def revoke_token(
        self,
        session: AsyncSession,
        token: str,
        reason: str = "manual_revocation",
        revoked_by: Optional[str] = None
    ) -> bool:
        """Revoke a specific token"""
        try:
            # Decode token to get JTI (without verification for revocation)
            payload = jwt.decode(token, options={"verify_signature": False})
            token_jti = payload["jti"]
            user_id = payload["sub"]
            token_type = payload.get("type", "access")
            
            # Add to revoked tokens
            revoked_token = RevokedToken(
                token_jti=token_jti,
                user_id=user_id,
                reason=reason,
                revoked_by=revoked_by,
                token_type=token_type,
                original_expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            )
            
            session.add(revoked_token)
            
            # Deactivate associated session
            if token_type == "refresh":
                await session.execute(
                    update(EnhancedUserSession)
                    .where(EnhancedUserSession.refresh_token_jti == token_jti)
                    .values(is_active=False)
                )
            
            # Log revocation
            await self.audit_logger.log_security_event(
                session=session,
                event_type=SecurityEventType.API_KEY_REVOKED,
                user_id=user_id,
                action="token_revoked",
                success=True,
                details={
                    "token_type": token_type,
                    "reason": reason,
                    "revoked_by": revoked_by
                }
            )
            
            await session.commit()
            return True
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def _enforce_session_limits(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        max_sessions: int = 3
    ):
        """Enforce maximum concurrent sessions per user"""
        try:
            # Get security configuration for org
            config_result = await session.execute(
                select(SecurityConfiguration).where(SecurityConfiguration.org_id == org_id)
            )
            config = config_result.scalar_one_or_none()
            
            if config:
                max_sessions = config.max_concurrent_sessions
            
            # Get active sessions for user
            active_sessions = await session.execute(
                select(EnhancedUserSession)
                .where(
                    EnhancedUserSession.user_id == user_id,
                    EnhancedUserSession.is_active == True
                )
                .order_by(EnhancedUserSession.last_activity.desc())
            )
            
            sessions = active_sessions.scalars().all()
            
            # If we're at the limit, deactivate oldest sessions
            if len(sessions) >= max_sessions:
                sessions_to_deactivate = sessions[max_sessions-1:]  # Keep newest sessions
                
                for old_session in sessions_to_deactivate:
                    await self._revoke_session_tokens(session, old_session, "session_limit_exceeded")
                    
                    await session.execute(
                        update(EnhancedUserSession)
                        .where(EnhancedUserSession.id == old_session.id)
                        .values(is_active=False)
                    )
            
        except Exception as e:
            logger.error(f"Failed to enforce session limits: {e}")
            raise
    
    async def _revoke_session_tokens(
        self,
        session: AsyncSession,
        user_session: EnhancedUserSession,
        reason: str
    ):
        """Revoke all tokens associated with a session"""
        try:
            # Revoke access token
            access_revoked = RevokedToken(
                token_jti=user_session.access_token_jti,
                user_id=user_session.user_id,
                reason=reason,
                token_type="access"
            )
            session.add(access_revoked)
            
            # Revoke refresh token
            refresh_revoked = RevokedToken(
                token_jti=user_session.refresh_token_jti,
                user_id=user_session.user_id,
                reason=reason,
                token_type="refresh"
            )
            session.add(refresh_revoked)
            
        except Exception as e:
            logger.error(f"Failed to revoke session tokens: {e}")
            raise

# Global enhanced JWT manager instance
_enhanced_jwt_manager: Optional[EnhancedJWTManager] = None

def get_enhanced_jwt_manager() -> EnhancedJWTManager:
    """Get the global enhanced JWT manager instance"""
    global _enhanced_jwt_manager
    if _enhanced_jwt_manager is None:
        _enhanced_jwt_manager = EnhancedJWTManager()
    return _enhanced_jwt_manager