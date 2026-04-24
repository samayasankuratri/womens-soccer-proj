import { useState } from "react";

function App() {
  const [video, setVideo] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!video) return;
    setLoading(true);
    setError(null);
    setOutputUrl(null);
    const formData = new FormData();
    formData.append("video", video);
    try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Analysis failed");
      const blob = await response.blob();
      setOutputUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError("Something went wrong. Make sure your backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{backgroundColor:"#0a0a0f",minHeight:"100vh",color:"white",fontFamily:"Arial,sans-serif"}}>
      <div style={{background:"linear-gradient(135deg,#0d1b2a,#1a1a3e)",padding:"20px 40px",borderBottom:"1px solid #ffffff15"}}>
        <h1 style={{margin:0,fontSize:"22px",fontWeight:"700",color:"#00d4ff"}}>ASA Soccer Analysis</h1>
        <p style={{margin:0,fontSize:"12px",color:"#ffffff60"}}>AI-Powered Player Tracking and Team Classification</p>
      </div>
      <div style={{maxWidth:"900px",margin:"0 auto",padding:"50px 20px"}}>
        <div style={{textAlign:"center",marginBottom:"50px"}}>
          <h2 style={{fontSize:"42px",fontWeight:"800",margin:"0 0 15px",color:"white"}}>Analyze Your Match</h2>
          <p style={{color:"#ffffff70",fontSize:"16px",margin:0}}>Upload a soccer clip to detect players, track the ball, and classify teams</p>
        </div>
        <div style={{display:"flex",justifyContent:"center",gap:"12px",marginBottom:"40px"}}>
          <span style={{backgroundColor:"#ffffff10",border:"1px solid #ffffff20",borderRadius:"20px",padding:"6px 16px",fontSize:"13px",color:"#ffffffcc"}}>Player Tracking</span>
          <span style={{backgroundColor:"#ffffff10",border:"1px solid #ffffff20",borderRadius:"20px",padding:"6px 16px",fontSize:"13px",color:"#ffffffcc"}}>Ball Detection</span>
          <span style={{backgroundColor:"#ffffff10",border:"1px solid #ffffff20",borderRadius:"20px",padding:"6px 16px",fontSize:"13px",color:"#ffffffcc"}}>Team Classification</span>
        </div>
        <div style={{border:"2px dashed #ffffff25",borderRadius:"16px",padding:"50px 30px",textAlign:"center",backgroundColor:"#ffffff05",marginBottom:"24px"}}>
          <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files[0])} style={{color:"white",marginBottom:"10px"}} />
          {video && <p style={{color:"#00d4ff",margin:"10px 0 0"}}>Selected: {video.name}</p>}
        </div>
        <button onClick={handleUpload} disabled={!video || loading} style={{width:"100%",padding:"16px",borderRadius:"12px",border:"none",fontSize:"16px",fontWeight:"700",cursor:"pointer",background:"linear-gradient(135deg,#00d4ff,#7b2ff7)",color:"white"}}>
          {loading ? "Analyzing... Please wait" : "Analyze Video"}
        </button>
        {error && <div style={{marginTop:"20px",padding:"16px",borderRadius:"12px",color:"#ff6b6b",textAlign:"center"}}>{error}</div>}
        {outputUrl && (
          <div style={{marginTop:"50px"}}>
            <h2 style={{textAlign:"center",color:"#00d4ff"}}>Analysis Complete</h2>
            <video src={outputUrl} controls style={{width:"100%",borderRadius:"16px"}} />
            <a href={outputUrl} download="analyzed_soccer.mp4" style={{display:"block",textAlign:"center",marginTop:"16px",padding:"14px",borderRadius:"12px",background:"linear-gradient(135deg,#00d4ff,#7b2ff7)",color:"white",fontWeight:"700",textDecoration:"none"}}>
              Download Analyzed Video
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
