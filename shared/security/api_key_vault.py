"""
Enterprise API Key Vault for LiftOS
Provides AES-256 encryption, secure storage, and key management for all API providers
"""

import os
import json
import base64
import secrets
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import logging

from ..database.models import EncryptedAPIKey, SecurityAuditLog
from .audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)

class APIKeyVault:
    """
    Enterprise-grade API Key Vault with AES-256-GCM encryption
    Follows Bank of England security standards
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize the API Key Vault"""
        self.master_key = master_key or self._get_master_key()
        self.audit_logger = SecurityAuditLogger()
        
        # Supported API providers
        self.supported_providers = {
            'meta_business', 'google_ads', 'klaviyo', 'shopify', 'woocommerce',
            'amazon', 'hubspot', 'salesforce', 'stripe', 'paypal', 'tiktok',
            'snowflake', 'databricks', 'zoho_crm', 'linkedin_ads', 'x_ads'
        }
    
    def _get_master_key(self) -> str:
        """Get or generate master encryption key"""
        master_key = os.getenv("API_VAULT_MASTER_KEY")
        if not master_key:
            # Generate a new master key for development
            master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            logger.warning("Generated new master key - store securely in production!")
        return master_key
    
    def _derive_key(self, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive encryption key using PBKDF2-HMAC-SHA256"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(self.master_key.encode())
    
    def _encrypt_data(self, data: str) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data using AES-256-GCM
        Returns: (encrypted_data, salt, iv)
        """
        # Generate random salt and IV
        salt = os.urandom(16)
        iv = os.urandom(12)  # GCM mode uses 96-bit IV
        
        # Derive key from master key and salt
        key = self._derive_key(salt)
        
        # Encrypt using AES-256-GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
        
        # Combine ciphertext and authentication tag
        encrypted_data = ciphertext + encryptor.tag
        
        return encrypted_data, salt, iv
    
    def _decrypt_data(self, encrypted_data: bytes, salt: bytes, iv: bytes) -> str:
        """
        Decrypt data using AES-256-GCM
        """
        # Derive key from master key and salt
        key = self._derive_key(salt)
        
        # Split ciphertext and authentication tag
        ciphertext = encrypted_data[:-16]
        tag = encrypted_data[-16:]
        
        # Decrypt using AES-256-GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext.decode()
    
    async def store_api_key(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        credentials: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store encrypted API key for a provider
        Returns the key ID
        """
        if provider not in self.supported_providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        try:
            # Serialize credentials
            credentials_json = json.dumps(credentials)
            
            # Encrypt credentials
            encrypted_data, salt, iv = self._encrypt_data(credentials_json)
            
            # Create database record
            api_key_record = EncryptedAPIKey(
                user_id=user_id,
                org_id=org_id,
                provider=provider,
                encrypted_key=base64.b64encode(encrypted_data).decode(),
                salt=base64.b64encode(salt).decode(),
                iv=base64.b64encode(iv).decode(),
                metadata=metadata or {},
                status='active',
                next_rotation=datetime.now(timezone.utc) + timedelta(days=90)
            )
            
            session.add(api_key_record)
            await session.flush()
            
            key_id = str(api_key_record.id)
            
            # Log security event
            await self.audit_logger.log_api_key_created(
                session=session,
                user_id=user_id,
                org_id=org_id,
                provider=provider,
                key_id=key_id
            )
            
            await session.commit()
            
            logger.info(f"API key stored for provider {provider}, user {user_id}")
            return key_id
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to store API key for {provider}: {e}")
            raise
    
    async def retrieve_api_key(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt API key for a provider
        """
        try:
            # Get the most recent active key for this provider
            result = await session.execute(
                select(EncryptedAPIKey)
                .where(
                    EncryptedAPIKey.user_id == user_id,
                    EncryptedAPIKey.org_id == org_id,
                    EncryptedAPIKey.provider == provider,
                    EncryptedAPIKey.status == 'active'
                )
                .order_by(EncryptedAPIKey.created_at.desc())
                .limit(1)
            )
            
            api_key_record = result.scalar_one_or_none()
            if not api_key_record:
                return None
            
            # Decrypt credentials
            encrypted_data = base64.b64decode(api_key_record.encrypted_key)
            salt = base64.b64decode(api_key_record.salt)
            iv = base64.b64decode(api_key_record.iv)
            
            credentials_json = self._decrypt_data(encrypted_data, salt, iv)
            credentials = json.loads(credentials_json)
            
            # Update last used timestamp
            await session.execute(
                update(EncryptedAPIKey)
                .where(EncryptedAPIKey.id == api_key_record.id)
                .values(
                    last_used=datetime.now(timezone.utc),
                    usage_count=EncryptedAPIKey.usage_count + 1
                )
            )
            
            # Log access event
            await self.audit_logger.log_api_key_accessed(
                session=session,
                user_id=user_id,
                org_id=org_id,
                provider=provider,
                key_id=str(api_key_record.id)
            )
            
            await session.commit()
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            raise
    
    async def rotate_api_key(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        new_credentials: Dict[str, Any]
    ) -> str:
        """
        Rotate API key - mark old as rotated and store new one
        """
        try:
            # Mark existing key as rotated
            await session.execute(
                update(EncryptedAPIKey)
                .where(
                    EncryptedAPIKey.user_id == user_id,
                    EncryptedAPIKey.org_id == org_id,
                    EncryptedAPIKey.provider == provider,
                    EncryptedAPIKey.status == 'active'
                )
                .values(
                    status='rotated',
                    rotated_at=datetime.now(timezone.utc)
                )
            )
            
            # Store new key
            new_key_id = await self.store_api_key(
                session=session,
                user_id=user_id,
                org_id=org_id,
                provider=provider,
                credentials=new_credentials,
                metadata={'rotation_reason': 'scheduled_rotation'}
            )
            
            # Log rotation event
            await self.audit_logger.log_api_key_rotated(
                session=session,
                user_id=user_id,
                org_id=org_id,
                provider=provider,
                old_key_id="rotated",
                new_key_id=new_key_id
            )
            
            logger.info(f"API key rotated for provider {provider}, user {user_id}")
            return new_key_id
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to rotate API key for {provider}: {e}")
            raise
    
    async def revoke_api_key(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        reason: str = "manual_revocation"
    ) -> bool:
        """
        Revoke API key immediately
        """
        try:
            result = await session.execute(
                update(EncryptedAPIKey)
                .where(
                    EncryptedAPIKey.user_id == user_id,
                    EncryptedAPIKey.org_id == org_id,
                    EncryptedAPIKey.provider == provider,
                    EncryptedAPIKey.status == 'active'
                )
                .values(
                    status='revoked',
                    revoked_at=datetime.now(timezone.utc),
                    revocation_reason=reason
                )
            )
            
            if result.rowcount > 0:
                # Log revocation event
                await self.audit_logger.log_api_key_revoked(
                    session=session,
                    user_id=user_id,
                    org_id=org_id,
                    provider=provider,
                    reason=reason
                )
                
                await session.commit()
                logger.info(f"API key revoked for provider {provider}, user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to revoke API key for {provider}: {e}")
            raise
    
    async def list_api_keys(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all API keys for a user/org (without decrypting credentials)
        """
        try:
            result = await session.execute(
                select(EncryptedAPIKey)
                .where(
                    EncryptedAPIKey.user_id == user_id,
                    EncryptedAPIKey.org_id == org_id
                )
                .order_by(EncryptedAPIKey.provider, EncryptedAPIKey.created_at.desc())
            )
            
            api_keys = result.scalars().all()
            
            key_list = []
            for key in api_keys:
                key_info = {
                    'id': str(key.id),
                    'provider': key.provider,
                    'status': key.status,
                    'created_at': key.created_at.isoformat(),
                    'last_used': key.last_used.isoformat() if key.last_used else None,
                    'usage_count': key.usage_count,
                    'next_rotation': key.next_rotation.isoformat() if key.next_rotation else None
                }
                
                if include_metadata:
                    key_info['metadata'] = key.metadata
                
                key_list.append(key_info)
            
            return key_list
            
        except Exception as e:
            logger.error(f"Failed to list API keys for user {user_id}: {e}")
            raise
    
    async def get_keys_due_for_rotation(
        self,
        session: AsyncSession,
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get API keys that are due for rotation within specified days
        """
        try:
            cutoff_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
            
            result = await session.execute(
                select(EncryptedAPIKey)
                .where(
                    EncryptedAPIKey.status == 'active',
                    EncryptedAPIKey.next_rotation <= cutoff_date
                )
                .order_by(EncryptedAPIKey.next_rotation)
            )
            
            keys_due = result.scalars().all()
            
            return [
                {
                    'id': str(key.id),
                    'user_id': key.user_id,
                    'org_id': key.org_id,
                    'provider': key.provider,
                    'next_rotation': key.next_rotation.isoformat(),
                    'days_until_rotation': (key.next_rotation - datetime.now(timezone.utc)).days
                }
                for key in keys_due
            ]
            
        except Exception as e:
            logger.error(f"Failed to get keys due for rotation: {e}")
            raise

# Global vault instance
_api_key_vault: Optional[APIKeyVault] = None

def get_api_key_vault() -> APIKeyVault:
    """Get the global API key vault instance"""
    global _api_key_vault
    if _api_key_vault is None:
        _api_key_vault = APIKeyVault()
    return _api_key_vault