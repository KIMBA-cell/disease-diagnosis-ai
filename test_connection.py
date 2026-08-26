import psycopg2

DATABASE_URL = "postgresql://postgres.nyxlvdwwwmxukexadrrl:Kimba%400010%40@aws-0-eu-west-2.pooler.supabase.com:5432/postgres"

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:")
    print(e)