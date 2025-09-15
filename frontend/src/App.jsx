import React, { useState, useEffect } from 'react';
import './index.css';
import './App.css';
import bgImage from './assets/bg.jpg';

const API_BASE = 'https://ottoman-kws-backend-19726578476.europe-west4.run.app';

const KEYWORDS = [
  "arak", "bağ", "boza", "bozahane",
  "duhan", "hamr", "mahzen", "meyhane",
  "şarap", "şıra", "üzüm"
];

const IMAGE_MAP = {
  arak:      "arak.jpg",
  "bağ":     "bağ.jpg",
  boza:      "boza.jpg",
  bozahane:  "bozahane.png",
  duhan:     "duhan.png",
  hamr:      "hamr.png",
  mahzen:    "mahzen.JPG",
  meyhane:   "meyhane.png",
  şarap:     "şarap.png",
  şıra:      "şıra.jpg",
  üzüm:      "üzüm.jpg",
};

const TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

async function createTinyImage() {
  const url = `data:image/png;base64,${TINY_PNG_B64}`;
  const resp = await fetch(url);
  const blob = await resp.blob();
  return new File([blob], 'probe.png', { type: 'image/png' });
}

async function parseJSON(res) {
  const txt = await res.text();
  try { return JSON.parse(txt); } catch { return { error: 'non-JSON', raw: txt }; }
}

async function findDocumentPath(docName) {
  const basePath = '/Documents';
  const extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'];

  //console.log(`Looking for document: ${docName}`);

  for (const ext of extensions) {
    const testPath = `${basePath}/${docName}${ext}`;
    try {
      const response = await fetch(testPath, { method: 'HEAD' });
      if (response.ok) {
        //console.log(`Found document at: ${testPath}`);
        return testPath;
      }
    } catch (e) {
    }
  }
  
  console.warn(`Document not found: ${docName}, using fallback`);
  console.warn(`Make sure the file exists at: public${basePath}/${docName}[.extension]`);
  return `${basePath}/${docName}`;
}

async function postWithTinyImage(url) {
  const file = await createTinyImage();
  const form = new FormData();
  form.append('image', file);
  const res = await fetch(url, { method: 'POST', body: form });
  return { res, data: await parseJSON(res) };
}

async function setupVectors() {
  const { res, data } = await postWithTinyImage(`${API_BASE}/match?inspect=1&debug=0`);
  if (!res.ok) return '';
  
  const indexInfo = data.index_info || {};
  
  return '';
}

async function searchDocuments(keyword, vectorName, searchMode = 'balanced') {
  const imgResp = await fetch(`/images/${IMAGE_MAP[keyword]}`, { cache: 'no-cache' });
  if (!imgResp.ok) throw new Error(`Could not load image for ${keyword}`);
  
  const blob = await imgResp.blob();
  const file = new File([blob], IMAGE_MAP[keyword], { type: blob.type || 'image/jpeg' });

  const form = new FormData();
  form.append('image', file);
  
  // First, get a large sample with a permissive threshold to analyze distance distribution
  const params = new URLSearchParams({
    threshold: '0.5', // Very permissive to get good sample
    debug: '0',
  });

  const res = await fetch(`${API_BASE}/match?${params}`, { method: 'POST', body: form });
  const data = await parseJSON(res);
  
  if (!res.ok) throw new Error(`Search failed: ${data.error || 'Unknown error'}`);
  
  const allMatches = data.matches || [];
  
  if (allMatches.length === 0) {
    return [];
  }
  
  // Analyze distance distribution
  const distances = allMatches.map(m => m.distance).sort((a, b) => a - b);
  const minDist = distances[0];
  const maxDist = distances[distances.length - 1];
  const medianDist = distances[Math.floor(distances.length / 2)];
  const p25 = distances[Math.floor(distances.length * 0.25)];
  const p75 = distances[Math.floor(distances.length * 0.75)];
  
  // Calculate adaptive thresholds based on distribution
  let threshold;
  let description;
  
  switch(searchMode) {
    case 'precise':
      // Very selective: only the closest ~20% of matches
      threshold = Math.min(p25, minDist + (maxDist - minDist) * 0.3);
      description = 'Most Similar Only';
      break;
    case 'expanded':
      // More expanded: ~80% of matches
      threshold = Math.min(p75, minDist + (maxDist - minDist) * 0.8);
      description = 'Broader Results';
      break;
    case 'balanced':
    default:
      // Balanced: around median, ~50% of matches
      threshold = Math.min(medianDist, minDist + (maxDist - minDist) * 0.6);
      description = 'Balanced Selection';
      break;
  }
  
  // Filter matches based on calculated threshold
  const filteredMatches = allMatches.filter(m => m.distance <= threshold);
  
  // Log comprehensive search results info
  console.log('=== ADAPTIVE SEARCH RESULTS ===');
  console.log(`Search mode: ${searchMode} (${description})`);
  console.log(`Distance range: ${minDist.toFixed(4)} to ${maxDist.toFixed(4)}`);
  console.log(`Calculated threshold: ${threshold.toFixed(4)}`);
  console.log(`Total candidates: ${allMatches.length}`);
  console.log(`Filtered matches: ${filteredMatches.length}`);
  console.log(`Distribution: min=${minDist.toFixed(4)}, p25=${p25.toFixed(4)}, median=${medianDist.toFixed(4)}, p75=${p75.toFixed(4)}, max=${maxDist.toFixed(4)}`);
  
  return filteredMatches;
}

async function processMatches(matches) {
  if (!matches?.length) return [];

  const groups = {};
  matches.forEach(match => {
    const docName = match.doc || 'unknown';
    if (!groups[docName]) {
      groups[docName] = { name: docName, matches: [], bestDistance: Infinity };
    }
    
    groups[docName].matches.push({
      distance: match.distance,
      score: match.score,
      coords: typeof match.coords === 'string' ? JSON.parse(match.coords) : (match.coords || [0, 0, 0, 0])
    });
    
    groups[docName].bestDistance = Math.min(groups[docName].bestDistance, match.distance);
  });

  const docs = Object.values(groups).sort((a, b) => a.bestDistance - b.bestDistance);

  for (const doc of docs) {
    doc.imagePath = await findDocumentPath(doc.name);
  }

  return docs;
}

async function createHighlightedDocument(imagePath, matches) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      
      //console.log(`Processing image: ${img.naturalWidth} x ${img.naturalHeight}`);
      
      ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
      
      const highlightCanvas = document.createElement('canvas');
      const highlightCtx = highlightCanvas.getContext('2d');
      highlightCanvas.width = img.naturalWidth;
      highlightCanvas.height = img.naturalHeight;
      
      matches.forEach((match, index) => {
        const [x, y, width, height] = match.coords;
        
        highlightCtx.globalCompositeOperation = 'source-over';
        highlightCtx.fillStyle = 'rgba(255, 255, 0, 0.4)';
        highlightCtx.fillRect(x, y, width, height);
      });
      
      ctx.globalCompositeOperation = 'multiply';
      ctx.drawImage(highlightCanvas, 0, 0);
      
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = 'rgba(255, 255, 0, 0.15)';
      matches.forEach((match) => {
        const [x, y, width, height] = match.coords;
        ctx.fillRect(x, y, width, height);
      });
      
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'rgba(255, 200, 0, 0.6)';
      ctx.lineWidth = 1;
      
      matches.forEach((match) => {
        const [x, y, width, height] = match.coords;
        ctx.strokeRect(x, y, width, height);
      });
      
      canvas.toBlob((blob) => {
        if (blob) {
          const highlightedImageUrl = URL.createObjectURL(blob);
          //console.log('Successfully created highlighted document');
          resolve(highlightedImageUrl);
        } else {
          reject(new Error('Failed to create blob from canvas'));
        }
      }, 'image/png', 1.0);
    };
    
    img.onerror = () => {
      reject(new Error(`Failed to load image: ${imagePath}`));
    };
    
    img.src = imagePath;
  });
}

function DocumentViewer({ document, onClose }) {
  const [highlightedImageUrl, setHighlightedImageUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  if (!document) return null;

  useEffect(() => {
    let isMounted = true;
    
    const processDocument = async () => {
      try {
        setLoading(true);
        setError(null);
        
        //console.log('Creating highlighted version of:', document.name);
        //console.log('Matches to embed:', document.matches.length);
        
        const highlightedUrl = await createHighlightedDocument(document.imagePath, document.matches);
        
        if (isMounted) {
          setHighlightedImageUrl(highlightedUrl);
          setLoading(false);
        }
      } catch (err) {
        console.error('Error creating highlighted document:', err);
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      }
    };
    
    processDocument();
    
    return () => {
      isMounted = false;
      if (highlightedImageUrl) {
        URL.revokeObjectURL(highlightedImageUrl);
      }
    };
  }, [document.imagePath, document.matches]);

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.9)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20
    }}>
      <div style={{
        backgroundColor: 'white', borderRadius: 8,
        maxWidth: '95vw', maxHeight: '95vh', overflow: 'auto'
      }}>
        <div style={{
          padding: 16, borderBottom: '1px solid #ddd',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          backgroundColor: '#f5f5f5'
        }}>
          <h3 style={{ margin: 0, color: '#000' }}>{document.name} (Highlighted Version)</h3>
          <div>
            <span style={{ marginRight: 16, fontSize: 14, color: '#666' }}>
              {document.matches.length} match{document.matches.length !== 1 ? 'es' : ''} embedded
            </span>
            <button onClick={onClose} style={{
              background: 'none', border: 'none', fontSize: 24, cursor: 'pointer', padding: 4, color: '#000'
            }}>×</button>
          </div>
        </div>

        <div style={{ padding: 16, textAlign: 'center' }}>
          {loading && (
            <div style={{
              padding: 40, color: '#666'
            }}>
              <div style={{ marginBottom: 16 }}>Creating highlighted version...</div>
              <div style={{ fontSize: 14 }}>Embedding {document.matches.length} highlights into document</div>
            </div>
          )}
          
          {error && (
            <div style={{
              padding: 40, color: '#c33', backgroundColor: '#fee', borderRadius: 4
            }}>
              <div>Error creating highlighted document:</div>
              <div style={{ fontSize: 14, marginTop: 8 }}>{error}</div>
            </div>
          )}
          
          {highlightedImageUrl && !loading && (
            <div>
              <img
                src={highlightedImageUrl}
                alt={`${document.name} with embedded highlights`}
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  border: '2px solid #28a745',
                  borderRadius: 4,
                  boxShadow: '0 4px 16px rgba(0,0,0,0.2)'
                }}
              />
            </div>
          )}
        </div>

        <div style={{ padding: 16, borderTop: '1px solid #ddd', backgroundColor: '#f9f9f9' }}>
          <h4 style={{ margin: '0 0 12px 0' }}>Embedded Highlights:</h4>
          {document.matches.map((match, i) => (
            <div key={i} style={{ marginBottom: 8, fontSize: 12 }}>
              <strong style={{ color: '#FF0000' }}>#{i + 1}</strong> 
              <span style={{ marginLeft: 8 }}>
                distance: {match.distance.toFixed(3)}, score: {match.score.toFixed(3)}
              </span>
              <br />
              <span style={{ color: '#666' }}>
                position: x={match.coords[0]}, y={match.coords[1]}, 
                size: {match.coords[2]}×{match.coords[3]}px
              </span>
            </div>
          ))}
          <div style={{ marginTop: 12, fontSize: 11, color: '#999', fontStyle: 'italic' }}>
            This is a temporary highlighted version. The original document is unchanged.
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [keyword, setKeyword] = useState("");
  const [searchMode, setSearchMode] = useState("balanced");
  const [preview, setPreview] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [vectorName, setVectorName] = useState("");

  useEffect(() => {
    setupVectors().then(name => {
      if (name) setVectorName(name);
    }).catch(console.error);
  }, []);

  const handleKeywordChange = (e) => {
    const kw = e.target.value;
    setKeyword(kw);
    setDocuments([]);
    setPreview(kw ? `/images/${IMAGE_MAP[kw]}` : null);
  };

  const handleSearch = async () => {
    if (!keyword) return;

    try {
      setLoading(true);
      setError("");
      setDocuments([]);

      const matches = await searchDocuments(keyword, vectorName, searchMode);
      const docs = await processMatches(matches);
      setDocuments(docs);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', padding: 20,
      backgroundImage: `url(${bgImage})`,
      backgroundSize: 'cover', backgroundPosition: 'center', backgroundAttachment: 'fixed'
    }}>
      <div style={{ 
        maxWidth: 1200, margin: '0 auto',
        backgroundColor: 'rgba(255,255,255,0.95)',
        borderRadius: 12, padding: 24, boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ 
          textAlign: 'center', marginBottom: 32, color: '#333',
          fontSize: 32, fontWeight: 'bold'
        }}>Tahrir Ottoman KWS</h1>

        <div style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'end', justifyContent: 'center', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 250 }}>
              <span style={{ fontWeight: 500, textAlign: 'center', color: '#000' }}>Select Keyword to Search</span>
              <select
                value={keyword}
                onChange={handleKeywordChange}
                style={{ padding: 12, borderRadius: 4, border: '1px solid #ddd', fontSize: 16 }}
              >
                <option value="">— Choose a keyword —</option>
                {KEYWORDS.map(kw => (
                  <option key={kw} value={kw}>{kw}</option>
                ))}
              </select>
            </label>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minWidth: 200 }}>
              <span style={{ fontWeight: 500, textAlign: 'center', color: '#000' }}>Search Precision</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 8, border: '1px solid #ddd', borderRadius: 4, backgroundColor: '#f9f9f9' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 14, color: '#000' }}>
                  <input
                    type="radio"
                    name="searchMode"
                    value="precise"
                    checked={searchMode === 'precise'}
                    onChange={(e) => setSearchMode(e.target.value)}
                    style={{ margin: 0 }}
                  />
                  <span>Most Similar Only</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 14, color: '#000' }}>
                  <input
                    type="radio"
                    name="searchMode"
                    value="balanced"
                    checked={searchMode === 'balanced'}
                    onChange={(e) => setSearchMode(e.target.value)}
                    style={{ margin: 0 }}
                  />
                  <span>Balanced Selection</span>
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 14, color: '#000' }}>
                  <input
                    type="radio"
                    name="searchMode"
                    value="expanded"
                    checked={searchMode === 'expanded'}
                    onChange={(e) => setSearchMode(e.target.value)}
                    style={{ margin: 0 }}
                  />
                  <span>Broader Results</span>
                </label>
              </div>
            </div>

            <button
              onClick={handleSearch}
              disabled={!keyword || loading}
              style={{ 
                padding: '12px 24px', 
                backgroundColor: keyword && !loading ? '#FF9800' : '#ccc', 
                color: 'white', border: 'none', borderRadius: 4,
                cursor: keyword && !loading ? 'pointer' : 'not-allowed',
                fontSize: 16, fontWeight: 500
              }}
            >
              {loading ? 'Searching...' : 'Search Documents'}
            </button>
          </div>
          
          <div style={{ textAlign: 'center', marginTop: 8, fontSize: 14, color: '#666' }}>
            Smart thresholds: automatically adapts to each keyword's similarity distribution
          </div>
        </div>

        {preview && (
          <div style={{ marginBottom: 24, textAlign: 'center' }}>
            <p style={{ fontWeight: 500, marginBottom: 8 }}>Searching for:</p>
            <img src={preview} alt={keyword} style={{ 
              maxHeight: 150, border: '2px solid #ddd', borderRadius: 8,
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }} />
          </div>
        )}

        {error && (
          <div style={{ 
            marginBottom: 24, padding: 12, backgroundColor: '#fee',
            border: '1px solid #fcc', borderRadius: 4, color: '#c33'
          }}>⚠️ {error}</div>
        )}

        {documents.length > 0 && (
          <>
            <h2 style={{ marginBottom: 20, color: '#333', fontSize: 24 }}>
              Top Documents ({documents.length} found)
            </h2>
            <div style={{ 
              display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 20 
            }}>
              {documents.map((doc, i) => (
                <div key={i} onClick={() => setSelectedDocument(doc)} style={{ 
                  border: '1px solid #ddd', borderRadius: 8, overflow: 'hidden',
                  backgroundColor: 'white', cursor: 'pointer', transition: 'all 0.2s ease',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ 
                    padding: 16, backgroundColor: '#f8f9fa', borderBottom: '1px solid #dee2e6' 
                  }}>
                    <h3 style={{ margin: '0 0 8px 0', fontSize: 16, fontWeight: 600, color: '#333' }}>
                      {doc.name}
                    </h3>
                    <div style={{ fontSize: 14, color: '#666' }}>
                      {doc.matches.length} match{doc.matches.length !== 1 ? 'es' : ''} • 
                      best: {doc.bestDistance.toFixed(3)}
                    </div>
                  </div>
                  
                  <div style={{ 
                    height: 200, backgroundColor: '#f0f0f0', display: 'flex',
                    alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden'
                  }}>
                    <img
                      src={doc.imagePath}
                      alt={doc.name}
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextElementSibling.style.display = 'flex';
                      }}
                    />
                    <div style={{
                      display: 'none', alignItems: 'center', justifyContent: 'center',
                      width: '100%', height: '100%', color: '#666', fontSize: 14
                    }}>Document Preview</div>
                    
                    <div style={{
                      position: 'absolute', top: 8, right: 8,
                      backgroundColor: '#ff4444', color: 'white', borderRadius: 12,
                      padding: '4px 8px', fontSize: 12, fontWeight: 600
                    }}>{doc.matches.length}</div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {loading && (
          <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
            <div style={{ fontSize: 18, marginBottom: 8 }}>Searching documents...</div>
            <div style={{ fontSize: 14 }}>This may take a moment</div>
          </div>
        )}

        {!loading && keyword && documents.length === 0 && !error && (
          <div style={{ 
            textAlign: 'center', padding: 40, color: '#666',
            backgroundColor: '#f8f9fa', borderRadius: 8
          }}>
            
          </div>
        )}
      </div>

      {selectedDocument && (
        <DocumentViewer
          document={selectedDocument}
          onClose={() => setSelectedDocument(null)}
        />
      )}
    </div>
  );
}