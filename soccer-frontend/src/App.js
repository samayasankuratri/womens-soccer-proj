import { useState, useRef } from "react";

function App() {
  const [video, setVideo] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [duelCount, setDuelCount] = useState(null);
  const [possession, setPossession] = useState(null);
  const [duelPositions, setDuelPositions] = useState([]);
  const fileInputRef = useRef(null);
  const progressInterval = useRef(null);

  const startFakeProgress = () => {
    const steps = [
      [5, "Uploading video..."],
      [15, "Initializing AI models..."],
      [30, "Collecting player crops..."],
      [50, "Training team classifier..."],
      [70, "Processing frames..."],
      [85, "Detecting players and ball..."],
      [92, "Finalizing analysis..."],
      [97, "Encoding output video..."],
    ];
    let i = 0;
    progressInterval.current = setInterval(() => {
      if (i < steps.length) {
        setProgress(steps[i][0]);
        setProgressLabel(steps[i][1]);
        i++;
      }
    }, 8000);
  };

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith("video/")) {
      setVideo(file);
      setVideoPreviewUrl(URL.createObjectURL(file));
      setOutputUrl(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!video) return;
    setLoading(true);
    setError(null);
    setOutputUrl(null);
    setProgress(0);
    setProgressLabel("Starting...");
    startFakeProgress();

    const formData = new FormData();
    formData.append("video", video);
    try {
      const response = await fetch("https://proxy-humbly-backless.ngrok-free.dev/analyze", {
        method: "POST",
        headers: { "ngrok-skip-browser-warning": "true" },
        body: formData,
      });
      if (!response.ok) throw new Error("Analysis failed");
      const blob = await response.blob();
      clearInterval(progressInterval.current);
      setProgress(100);
      setProgressLabel("Complete!");
      setDuelCount(response.headers.get('X-Duel-Count'));
      setOutputUrl(URL.createObjectURL(blob));
    } catch (err) {
      clearInterval(progressInterval.current);
      setError("Something went wrong. Make sure your backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files[0]);
  };

  const reset = () => {
    setVideo(null);
    setVideoPreviewUrl(null);
    setOutputUrl(null);
    setError(null);
    setProgress(0);
    setProgressLabel("");
    setDuelCount(null);
    setPossession(null);
    setDuelPositions([]);
  };

  return (
    <div style={{margin:0,padding:0,backgroundColor:"#050810",minHeight:"100vh",color:"white",fontFamily:"'Segoe UI',Arial,sans-serif"}}>

      {/* Header */}
      <div style={{background:"linear-gradient(90deg,#0d1b2a,#0a0f1e)",padding:"16px 40px",display:"flex",alignItems:"center",justifyContent:"space-between",borderBottom:"1px solid #ffffff10",position:"sticky",top:0,zIndex:100}}>
        <div style={{display:"flex",alignItems:"center",gap:"12px"}}>
          <div style={{width:"36px",height:"36px",borderRadius:"8px",background:"linear-gradient(135deg,#00d4ff,#7b2ff7)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"18px"}}>⚽</div>
          <div>
            <div style={{fontSize:"16px",fontWeight:"700",color:"white"}}>ASA Soccer Analysis</div>
            <div style={{fontSize:"11px",color:"#ffffff50"}}>AI-Powered Match Intelligence</div>
          </div>
        </div>
        <div style={{padding:"6px 14px",borderRadius:"20px",backgroundColor:"#ffffff08",border:"1px solid #ffffff15",fontSize:"12px",color:"#00d4ff"}}>Beta</div>
      </div>

      {/* Hero */}
      <div style={{textAlign:"center",padding:"60px 20px 40px",background:"radial-gradient(ellipse at top,#0d1b3e,#050810)"}}>
        <div style={{display:"inline-block",padding:"6px 16px",borderRadius:"20px",backgroundColor:"#00d4ff15",border:"1px solid #00d4ff30",fontSize:"12px",color:"#00d4ff",marginBottom:"20px",letterSpacing:"1px"}}>
          POWERED BY YOLO + COMPUTER VISION
        </div>
        <h1 style={{fontSize:"52px",fontWeight:"900",margin:"0 0 16px",lineHeight:1.1,background:"linear-gradient(135deg,#ffffff 0%,#00d4ff 50%,#7b2ff7 100%)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
          Analyze Your<br/>Soccer Match
        </h1>
        <p style={{color:"#ffffff60",fontSize:"16px",margin:"0 auto 30px",maxWidth:"500px"}}>
          Upload a clip and get AI-powered player tracking, ball detection, and team classification.
        </p>
        <div style={{display:"flex",justifyContent:"center",gap:"16px",flexWrap:"wrap"}}>
          {[["🏃","Player Tracking"],["⚽","Ball Detection"],["👕","Team Classification"]].map(([icon,title])=>(
            <div key={title} style={{backgroundColor:"#ffffff08",border:"1px solid #ffffff12",borderRadius:"10px",padding:"12px 18px",display:"flex",alignItems:"center",gap:"8px"}}>
              <span>{icon}</span>
              <span style={{fontSize:"13px",fontWeight:"600"}}>{title}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{maxWidth:"900px",margin:"0 auto",padding:"0 20px 80px"}}>

        {/* Upload Box */}
        {!outputUrl && (
          <div
            onDragOver={(e)=>{e.preventDefault();setDragOver(true);}}
            onDragLeave={()=>setDragOver(false)}
            onDrop={handleDrop}
            onClick={()=>!video && fileInputRef.current.click()}
            style={{border:`2px dashed ${dragOver?"#00d4ff":video?"#7b2ff7":"#ffffff20"}`,borderRadius:"20px",padding:"40px 30px",textAlign:"center",backgroundColor:dragOver?"#00d4ff08":video?"#7b2ff708":"#ffffff03",transition:"all 0.3s ease",marginBottom:"16px",cursor:video?"default":"pointer"}}
          >
            <input ref={fileInputRef} type="file" accept="video/*" onChange={(e)=>handleFileSelect(e.target.files[0])} style={{display:"none"}} />
            {video ? (
              <div>
                <p style={{color:"#00d4ff",fontWeight:"700",fontSize:"16px",margin:"0 0 4px"}}>✓ {video.name}</p>
                <p style={{color:"#ffffff40",fontSize:"13px",margin:"0 0 12px"}}>{(video.size/1024/1024).toFixed(1)} MB</p>
                <button onClick={(e)=>{e.stopPropagation();fileInputRef.current.click();}} style={{padding:"8px 16px",borderRadius:"8px",border:"1px solid #ffffff20",backgroundColor:"#ffffff10",color:"white",cursor:"pointer",fontSize:"13px"}}>Change File</button>
              </div>
            ) : (
              <div>
                <div style={{fontSize:"40px",marginBottom:"12px",opacity:0.6}}>🎬</div>
                <p style={{color:"white",fontWeight:"600",fontSize:"16px",margin:"0 0 6px"}}>Drop your video here</p>
                <p style={{color:"#ffffff40",fontSize:"13px",margin:0}}>or click to browse — MP4 recommended</p>
              </div>
            )}
          </div>
        )}

        {/* Video Preview */}
        {videoPreviewUrl && !outputUrl && (
          <div style={{marginBottom:"16px"}}>
            <p style={{color:"#ffffff60",fontSize:"13px",marginBottom:"8px",letterSpacing:"0.5px"}}>PREVIEW</p>
            <div style={{borderRadius:"12px",overflow:"hidden",border:"1px solid #ffffff15"}}>
              <video src={videoPreviewUrl} controls style={{width:"100%",display:"block",backgroundColor:"#000",maxHeight:"300px"}} />
            </div>
          </div>
        )}

        {/* Analyze Button */}
        {!outputUrl && (
          <button
            onClick={handleUpload}
            disabled={!video||loading}
            style={{width:"100%",padding:"18px",borderRadius:"14px",border:"none",fontSize:"16px",fontWeight:"700",cursor:!video||loading?"not-allowed":"pointer",background:!video||loading?"#ffffff10":"linear-gradient(135deg,#00d4ff,#7b2ff7)",color:!video||loading?"#ffffff30":"white",transition:"all 0.3s ease",boxShadow:!video||loading?"none":"0 0 30px #00d4ff30"}}
          >
            {loading?"Analyzing...":"▶  Analyze Video"}
          </button>
        )}

        {/* Progress Bar */}
        {loading && (
          <div style={{marginTop:"20px"}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:"8px"}}>
              <span style={{fontSize:"13px",color:"#ffffff70"}}>{progressLabel}</span>
              <span style={{fontSize:"13px",color:"#00d4ff",fontWeight:"600"}}>{progress}%</span>
            </div>
            <div style={{backgroundColor:"#ffffff08",borderRadius:"8px",overflow:"hidden",height:"8px"}}>
              <div style={{height:"100%",background:"linear-gradient(90deg,#00d4ff,#7b2ff7)",width:`${progress}%`,transition:"width 1s ease",borderRadius:"8px"}} />
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:"8px",marginTop:"16px"}}>
              {[["Upload","5%"],["AI Models","30%"],["Processing","70%"],["Complete","100%"]].map(([label,pct])=>(
                <div key={label} style={{textAlign:"center",padding:"8px",borderRadius:"8px",backgroundColor:progress>=parseInt(pct)?"#00d4ff15":"#ffffff05",border:`1px solid ${progress>=parseInt(pct)?"#00d4ff30":"#ffffff10"}`}}>
                  <div style={{fontSize:"11px",color:progress>=parseInt(pct)?"#00d4ff":"#ffffff40",fontWeight:"600"}}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div style={{marginTop:"16px",backgroundColor:"#ff000010",border:"1px solid #ff000030",borderRadius:"12px",padding:"16px",color:"#ff6b6b",textAlign:"center",fontSize:"14px"}}>
            ⚠️ {error}
          </div>
        )}

        {/* Stats Panel — always visible */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"12px",marginTop:"24px",marginBottom:"8px"}}>
          <div style={{padding:"20px",borderRadius:"16px",background:"#FF149315",border:"1px solid #FF149340",textAlign:"center"}}>
            <div style={{fontSize:"11px",color:"#FF1493",letterSpacing:"1px",fontWeight:"600",marginBottom:"8px"}}>TEAM A POSSESSION</div>
            <div style={{fontSize:"40px",fontWeight:"900",color:"#FF1493"}}>
              {possession ? `${possession.teamA}%` : "—"}
            </div>
          </div>
          <div style={{padding:"20px",borderRadius:"16px",background:"linear-gradient(135deg,#00d4ff15,#7b2ff715)",border:"1px solid #00d4ff30",textAlign:"center"}}>
            <div style={{fontSize:"11px",color:"#00d4ff",letterSpacing:"1px",fontWeight:"600",marginBottom:"8px"}}>DUELS DETECTED</div>
            <div style={{fontSize:"40px",fontWeight:"900",color:"#00d4ff"}}>
              {duelCount !== null ? duelCount : "—"}
            </div>
          </div>
          <div style={{padding:"20px",borderRadius:"16px",background:"#00BFFF15",border:"1px solid #00BFFF40",textAlign:"center"}}>
            <div style={{fontSize:"11px",color:"#00BFFF",letterSpacing:"1px",fontWeight:"600",marginBottom:"8px"}}>TEAM B POSSESSION</div>
            <div style={{fontSize:"40px",fontWeight:"900",color:"#00BFFF"}}>
              {possession ? `${possession.teamB}%` : "—"}
            </div>
          </div>
        </div>

        {/* Duel Heatmap — always visible */}
        <div style={{marginTop:"16px",marginBottom:"8px",padding:"20px",borderRadius:"16px",background:"#ffffff05",border:"1px solid #ffffff12"}}>
          <div style={{fontSize:"11px",color:"#00d4ff",letterSpacing:"1px",fontWeight:"600",marginBottom:"16px"}}>DUEL HEATMAP</div>
          <div style={{position:"relative",width:"100%"}}>
            <svg viewBox="0 0 100 65" style={{width:"100%",borderRadius:"8px",background:"#1a3a1a",display:"block"}}>
              {/* Field outline */}
              <rect x="1" y="1" width="98" height="63" fill="none" stroke="#ffffff25" strokeWidth="0.5"/>
              {/* Center line */}
              <line x1="50" y1="1" x2="50" y2="64" stroke="#ffffff25" strokeWidth="0.5"/>
              {/* Center circle */}
              <circle cx="50" cy="32.5" r="9" fill="none" stroke="#ffffff25" strokeWidth="0.5"/>
              <circle cx="50" cy="32.5" r="0.8" fill="#ffffff25"/>
              {/* Left penalty box */}
              <rect x="1" y="16" width="16" height="33" fill="none" stroke="#ffffff25" strokeWidth="0.5"/>
              {/* Left goal box */}
              <rect x="1" y="24" width="6" height="17" fill="none" stroke="#ffffff25" strokeWidth="0.5"/>
              {/* Right penalty box */}
              <rect x="83" y="16" width="16" height="33" fill="none" stroke="#ffffff25" strokeWidth="0.5"/>
              {/* Right goal box */}
              <rect x="93" y="24" width="6" height="17" fill="none" stroke="#ffffff25" strokeWidth="0.5"/>
              {/* Left penalty spot */}
              <circle cx="11" cy="32.5" r="0.8" fill="#ffffff25"/>
              {/* Right penalty spot */}
              <circle cx="89" cy="32.5" r="0.8" fill="#ffffff25"/>
              {/* Duel positions */}
              {duelPositions.map((pos, i) => (
                <circle key={i} cx={pos.x} cy={pos.y} r="3" fill="#00d4ff" fillOpacity="0.5" stroke="#00d4ff" strokeWidth="0.5"/>
              ))}
              {/* Placeholder text when no data */}
              {duelPositions.length === 0 && (
                <text x="50" y="35" textAnchor="middle" fill="#ffffff30" fontSize="4" fontFamily="Arial">
                  Run analysis to generate heatmap
                </text>
              )}
            </svg>
          </div>
        </div>

        {/* Side by Side Comparison */}
        {outputUrl && (
          <div style={{marginTop:"20px"}}>
            <div style={{display:"flex",alignItems:"center",gap:"12px",marginBottom:"24px"}}>
              <div style={{height:"1px",flex:1,background:"linear-gradient(90deg,transparent,#ffffff20)"}} />
              <span style={{fontSize:"14px",color:"#00d4ff",fontWeight:"600",letterSpacing:"1px"}}>ANALYSIS COMPLETE</span>
              <div style={{height:"1px",flex:1,background:"linear-gradient(90deg,#ffffff20,transparent)"}} />
            </div>

            {/* Videos */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"16px",marginBottom:"20px"}}>
              <div>
                <p style={{color:"#ffffff50",fontSize:"12px",letterSpacing:"1px",marginBottom:"8px"}}>ORIGINAL</p>
                <div style={{borderRadius:"12px",overflow:"hidden",border:"1px solid #ffffff15"}}>
                  <video src={videoPreviewUrl} controls style={{width:"100%",display:"block",backgroundColor:"#000"}} />
                </div>
              </div>
              <div>
                <p style={{color:"#00d4ff",fontSize:"12px",letterSpacing:"1px",marginBottom:"8px"}}>ANALYZED ✓</p>
                <div style={{borderRadius:"12px",overflow:"hidden",border:"1px solid #00d4ff30",boxShadow:"0 0 30px #00d4ff15"}}>
                  <video src={outputUrl} controls style={{width:"100%",display:"block",backgroundColor:"#000"}} />
                </div>
              </div>
            </div>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"12px"}}>
              <a href={outputUrl} download="analyzed_soccer.mp4" style={{display:"block",textAlign:"center",padding:"14px",borderRadius:"12px",background:"linear-gradient(135deg,#00d4ff,#7b2ff7)",color:"white",fontWeight:"700",textDecoration:"none",fontSize:"14px"}}>
                ⬇ Download Analyzed Video
              </a>
              <button onClick={reset} style={{padding:"14px",borderRadius:"12px",border:"1px solid #ffffff20",backgroundColor:"#ffffff08",color:"white",fontWeight:"700",cursor:"pointer",fontSize:"14px"}}>
                ↩ Analyze Another Video
              </button>
            </div>
          </div>
        )}
      </div>

      <style>{`* { box-sizing: border-box; }`}</style>
    </div>
  );
}

export default App;
