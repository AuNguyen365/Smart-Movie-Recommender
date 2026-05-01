import { useState, useEffect } from 'react'
import axios from 'axios'
import { Film, User, Link, Activity, Zap, Star, Search, Users } from 'lucide-react'
import './index.css'

function App() {
  const [metrics, setMetrics] = useState(null)
  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState('')
  const [recommendations, setRecommendations] = useState([])
  const [rules, setRules] = useState([])
  const [clusters, setClusters] = useState([])
  const [currentUserClusterId, setCurrentUserClusterId] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('recommendations')
  
  const [searchQuery, setSearchQuery] = useState('')
  const [searchRecs, setSearchRecs] = useState([])
  const [searchedMovie, setSearchedMovie] = useState(null)
  const [isSearching, setIsSearching] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    setSearchedMovie(null);
    try {
      const res = await axios.get(`http://localhost:8000/api/movie_recommendations?movie=${encodeURIComponent(searchQuery)}`)
      setSearchRecs(res.data.recommendations)
      setSearchedMovie(res.data.searched_movie)
    } catch (err) {
      console.error("Error searching movies:", err)
    } finally {
      setIsSearching(false)
    }
  }

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [metricsRes, usersRes, rulesRes, clustersRes] = await Promise.all([
          axios.get('http://localhost:8000/api/metrics'),
          axios.get('http://localhost:8000/api/users'),
          axios.get('http://localhost:8000/api/rules'),
          axios.get('http://localhost:8000/api/clusters')
        ])
        
        setMetrics(metricsRes.data)
        setUsers(usersRes.data.users)
        setRules(rulesRes.data.rules)
        setClusters(clustersRes.data.clusters)
        
        if (usersRes.data.users.length > 0) {
          setSelectedUser(usersRes.data.users[0])
        }
        setLoading(false)
      } catch (err) {
        console.error("Error fetching data:", err)
        setLoading(false)
      }
    }
    
    fetchInitialData()
  }, [])

  useEffect(() => {
    if (selectedUser) {
      const fetchRecommendations = async () => {
        try {
          const res = await axios.get(`http://localhost:8000/api/recommendations/${encodeURIComponent(selectedUser)}`)
          setRecommendations(res.data.recommendations)
          setCurrentUserClusterId(res.data.cluster_id)
        } catch (err) {
          console.error("Error fetching recommendations:", err)
        }
      }
      fetchRecommendations()
    }
  }, [selectedUser])

  if (loading) {
    return (
      <div className="loading-spinner">
        <div className="spinner"></div>
      </div>
    )
  }

  return (
    <>
      <div className="header">
        <h1>Smart Movie Recommender</h1>
        <p>Explore personalized movie suggestions powered by advanced clustering, association rules, and recommendation engines.</p>
      </div>

      {metrics && (
        <div className="dashboard-grid">
          <div className="glass-panel metric-card">
            <div className="metric-icon">
              <Film size={24} />
            </div>
            <div className="metric-info">
              <h3>Itemsets</h3>
              <p>{metrics.num_itemsets?.toLocaleString() || 0}</p>
            </div>
          </div>
          <div className="glass-panel metric-card">
            <div className="metric-icon">
              <Link size={24} />
            </div>
            <div className="metric-info">
              <h3>Association Rules</h3>
              <p>{metrics.num_rules?.toLocaleString() || 0}</p>
            </div>
          </div>
          <div className="glass-panel metric-card">
            <div className="metric-icon">
              <Zap size={24} />
            </div>
            <div className="metric-info">
              <h3>Avg. Lift</h3>
              <p>{metrics.avg_lift?.toFixed(2) || 0}</p>
            </div>
          </div>
        </div>
      )}

      <div className="main-content">
        <div className="sidebar">
          <div className="glass-panel">
            <h2 className="section-title"><User size={20} /> Select User</h2>
            <div className="user-selector">
              <select 
                className="custom-select"
                value={selectedUser}
                onChange={(e) => setSelectedUser(e.target.value)}
              >
                {users.map(user => (
                  <option key={user} value={user}>{user}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="glass-panel">
            <h2 className="section-title"><Activity size={20} /> System Status</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Status</span>
                <span style={{ color: '#4ade80' }}>Online</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Avg Confidence</span>
                <span>{(metrics?.avg_confidence * 100)?.toFixed(1)}%</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Mining Time</span>
                <span>{metrics?.mining_time_seconds?.toFixed(2)}s</span>
              </div>
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ minHeight: '500px' }}>
          <div className="tabs">
            <button 
              className={`tab-btn ${activeTab === 'recommendations' ? 'active' : ''}`}
              onClick={() => setActiveTab('recommendations')}
            >
              Top Recommendations
            </button>
            <button 
              className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              Movie Search
            </button>
            <button 
              className={`tab-btn ${activeTab === 'clusters' ? 'active' : ''}`}
              onClick={() => setActiveTab('clusters')}
            >
              Audience Clusters
            </button>
          </div>

          {activeTab === 'recommendations' && (
            <div className="movie-list">
              {recommendations.length > 0 ? (
                recommendations.map((rec, index) => (
                  <div key={index} className="movie-item">
                    <div className={`movie-rank ${index < 3 ? 'top-3' : ''}`}>
                      {rec.rank}
                    </div>
                    <div className="movie-details">
                      <div className="movie-title">{rec.movie_title}</div>
                      <div className="movie-score">
                        <Star size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
                        Score: {rec.score.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  <Film size={48} />
                  <h3>No recommendations found</h3>
                  <p>Select another user to see suggestions.</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'search' && (
            <div className="search-container">
              <form onSubmit={handleSearch} style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ flexGrow: 1, position: 'relative' }}>
                  <Search size={20} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                  <input 
                    type="text" 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter a movie name (e.g. Dune)..." 
                    style={{ width: '100%', padding: '0.75rem 1rem 0.75rem 3rem', borderRadius: '0.5rem', border: '1px solid var(--border-color)', background: 'rgba(15, 23, 42, 0.8)', color: 'var(--text-primary)', outline: 'none', fontSize: '1rem' }}
                  />
                </div>
                <button type="submit" style={{ padding: '0.75rem 1.5rem', borderRadius: '0.5rem', background: 'var(--accent-1)', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '1rem' }}>
                  {isSearching ? 'Searching...' : 'Search'}
                </button>
              </form>

              <div className="movie-list">
                {searchedMovie && (
                  <div className="movie-item searched-result" style={{ border: '1px solid var(--accent-1)', background: 'rgba(59, 130, 246, 0.1)', marginBottom: '2rem' }}>
                    <div className="movie-rank top-3" style={{ fontSize: '0.8rem' }}>Result</div>
                    <div className="movie-details">
                      <div className="movie-title" style={{ color: 'var(--accent-1)', fontSize: '1.25rem' }}>{searchedMovie}</div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Phim bạn vừa tìm kiếm</div>
                    </div>
                  </div>
                )}

                {searchRecs.length > 0 ? (
                  <>
                    <h3 className="section-title" style={{ marginBottom: '1rem', opacity: 0.8 }}><Zap size={18} /> Phim gợi ý cho bạn</h3>
                    {searchRecs.map((rec, index) => (
                      <div key={index} className="movie-item">
                        <div className={`movie-rank ${index < 3 ? 'top-3' : ''}`}>
                          {index + 1}
                        </div>
                        <div className="movie-details">
                          <div className="movie-title">{rec.movie_title}</div>
                          <div className="movie-score" style={{ display: 'flex', gap: '1rem', background: 'transparent', padding: 0 }}>
                            <span style={{ fontSize: '0.75rem', color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.15)', padding: '0.125rem 0.5rem', borderRadius: '1rem' }}>
                              <Zap size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
                              Lift: {rec.lift.toFixed(2)}x
                            </span>
                            <span style={{ fontSize: '0.75rem', color: '#93c5fd', background: 'rgba(59, 130, 246, 0.15)', padding: '0.125rem 0.5rem', borderRadius: '1rem' }}>
                              <Activity size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
                              Confidence: {(rec.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </>
                ) : (
                  !isSearching && searchQuery && (
                    <div className="empty-state">
                      <Film size={48} />
                      <h3>No related movies found</h3>
                      <p>We couldn't find any associations for this movie yet.</p>
                    </div>
                  )
                )}
                {!searchQuery && (
                  <div className="empty-state">
                    <Film size={48} />
                    <h3>Enter a movie name</h3>
                    <p>Find what movies are often watched together with your favorites.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'clusters' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div className="dashboard-grid">
                {clusters.map((cluster) => {
                  const isUserCluster = cluster.cluster_id === currentUserClusterId;
                  return (
                <div key={cluster.cluster_id} className="glass-panel" style={{ 
                  padding: '1.25rem', 
                  border: isUserCluster ? '2px solid var(--accent-1)' : '1px solid var(--border-color)', 
                  transform: isUserCluster ? 'scale(1.03)' : 'none', 
                  boxShadow: isUserCluster ? '0 0 25px rgba(59, 130, 246, 0.5)' : '',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
                    <h3 style={{ margin: 0, color: isUserCluster ? 'var(--accent-1)' : 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      Cluster {cluster.cluster_id}
                      {isUserCluster && <span style={{fontSize: '0.65rem', background: 'var(--accent-1)', color: '#fff', padding: '0.2rem 0.5rem', borderRadius: '1rem'}}>Current User</span>}
                    </h3>
                    <div className="badge"><Users size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />{cluster.num_users} users</div>
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Top Favorite Genres</div>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                      {cluster.top_genres.map(g => (
                        <span key={g} style={{ fontSize: '0.75rem', background: isUserCluster ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.05)', padding: '0.25rem 0.5rem', borderRadius: '4px', border: isUserCluster ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid rgba(255,255,255,0.1)' }}>{g}</span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Average Rating Given</div>
                    <div style={{ fontWeight: '600' }}>{cluster.avg_rating.toFixed(2)} / 10</div>
                  </div>
                </div>
              )})}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App
