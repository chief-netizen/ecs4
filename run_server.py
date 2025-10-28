from http.server import SimpleHTTPRequestHandler, HTTPServer
import whisper

print("Loading Whisper model (medium)... this may take a minute...")
model = whisper.load_model("base")
print("✅ Whisper model loaded.")

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/run':
            from your_script import main
            try:
                main(model)
                msg = "✅ Whisper + Argos + VITS ran successfully!"
            except Exception as e:
                msg = f"❌ Error: {e}"

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(msg.encode())

        elif self.path == '/kitten':
            from kitten_script import run_kitten
            try:
                run_kitten(model)
                msg = "✅ Whisper + KittenTTS ran successfully!"
            except Exception as e:
                msg = f"❌ Error: {e}"

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(msg.encode())

        else:
            super().do_GET()

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), Handler)
    print("Serving on http://localhost:8000")
    server.serve_forever()
