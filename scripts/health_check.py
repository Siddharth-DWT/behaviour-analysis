#!/usr/bin/env python3
"""
NEXUS Health Check - Verify all infrastructure services are running.
Run: python scripts/health_check.py
"""
import os
import sys
import json

def check_postgres():
    """Check PostgreSQL + pgvector connection."""
    try:
        import psycopg2
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            print("  [FAIL] DATABASE_URL environment variable not set")
            return False
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check pgvector extension
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        has_vector = cur.fetchone()
        
        # Check tables exist
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        # Check rule_config has seed data
        cur.execute("SELECT COUNT(*) FROM rule_config;")
        rule_count = cur.fetchone()[0]
        
        # Check schema version
        cur.execute("SELECT version, description FROM schema_version ORDER BY version DESC LIMIT 1;")
        version = cur.fetchone()
        
        conn.close()
        
        print(f"  [OK] PostgreSQL connected")
        print(f"  [OK] pgvector extension: {'installed' if has_vector else 'MISSING'}")
        print(f"  [OK] Tables created: {len(tables)} tables")
        print(f"       {', '.join(tables)}")
        print(f"  [OK] Rule config entries: {rule_count}")
        print(f"  [OK] Schema version: v{version[0]} - {version[1]}")
        return True
        
    except ImportError:
        print("  [WARN] psycopg2 not installed. Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"  [FAIL] PostgreSQL: {e}")
        return False

def check_redis():
    """Check Redis/Valkey connection and streams capability."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        
        # Basic ping
        r.ping()
        
        # Test stream operations
        test_stream = "nexus:health_check"
        msg_id = r.xadd(test_stream, {"test": "ok", "timestamp": "12345"})
        messages = r.xrange(test_stream, count=1)
        r.delete(test_stream)
        
        # Check info
        info = r.info("server")
        server_version = info.get("redis_version", info.get("valkey_version", "unknown"))
        
        print(f"  [OK] Redis/Valkey connected (version: {server_version})")
        print(f"  [OK] Streams working (test write/read successful)")
        return True
        
    except ImportError:
        print("  [WARN] redis-py not installed. Run: pip install redis")
        return False
    except Exception as e:
        print(f"  [FAIL] Redis: {e}")
        return False

def check_env():
    """Check environment variables."""
    import os
    
    issues = []
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    if not anthropic_key or anthropic_key.startswith("your_"):
        issues.append("ANTHROPIC_API_KEY not set (needed for LANGUAGE Agent + FUSION narratives)")
    else:
        print(f"  [OK] ANTHROPIC_API_KEY set (starts with {anthropic_key[:10]}...)")
    
    if not openai_key or openai_key.startswith("your_"):
        issues.append("OPENAI_API_KEY not set (needed for Whisper transcription + embeddings)")
    else:
        print(f"  [OK] OPENAI_API_KEY set (starts with {openai_key[:10]}...)")
    
    for issue in issues:
        print(f"  [WARN] {issue}")
    
    return len(issues) == 0


def main():
    print("=" * 60)
    print("NEXUS Health Check")
    print("=" * 60)
    
    all_ok = True
    
    print("\n1. PostgreSQL + pgvector:")
    if not check_postgres():
        all_ok = False
    
    print("\n2. Redis / Valkey:")
    if not check_redis():
        all_ok = False
    
    print("\n3. Environment Variables:")
    if not check_env():
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED - Infrastructure ready for development")
    else:
        print("SOME CHECKS FAILED - Fix issues above before developing")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
