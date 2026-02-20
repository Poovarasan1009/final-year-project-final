"""
Authentication and session management
"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import secrets

SECRET_KEY = "development_secret_key_fixed_for_debugging_12345"  # Fixed key for dev
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, 
                    detail="Invalid authentication scheme."
                )
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, 
                    detail="Invalid or expired token."
                )
            return credentials.credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="Invalid authorization code."
            )

    def verify_jwt(self, jwtoken: str) -> bool:
        try:
            payload = decode_jwt(jwtoken)
            return payload is not None
        except:
            return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("DEBUG: Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError as e:
        print(f"DEBUG: JWT Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def get_current_user(request: Request):
    """Get current user from session cookie"""
    token = request.cookies.get("access_token")
    print(f"DEBUG: get_current_user called. Token found: {token[:10] if token else 'None'}...")
    if token:
        try:
            payload = decode_jwt(token)
            print(f"DEBUG: Token decoded successfully. User: {payload.get('username')}")
            return payload
        except Exception as e:
            print(f"DEBUG: Token validation failed in get_current_user: {e}")
            return None
    print("DEBUG: No token found in cookies")
    return None

def check_role_access(user_role: str, required_role: str) -> bool:
    """Check if user has required role access"""
    role_hierarchy = {
        'admin': ['admin', 'teacher', 'student'],
        'teacher': ['teacher', 'student'],
        'student': ['student']
    }
    
    if user_role in role_hierarchy and required_role in role_hierarchy[user_role]:
        return True
    return False