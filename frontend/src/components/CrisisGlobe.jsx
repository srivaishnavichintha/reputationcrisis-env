import { useEffect, useRef } from "react";
import * as THREE from "three";
import { motion } from "framer-motion";

const CRISIS_COLORS = {
  LOW: { globe: 0x1a3a5c, hotspot: 0x00d4ff, glow: "#00d4ff" },
  MEDIUM: { globe: 0x3a2a1a, hotspot: 0xff8c00, glow: "#ff8c00" },
  HIGH: { globe: 0x3a0a0a, hotspot: 0xff1a1a, glow: "#ff1a1a" },
};

const HOTSPOT_POSITIONS = [
  { lat: 40.7, lon: -74.0 },  // New York
  { lat: 51.5, lon: -0.1 },   // London
  { lat: 35.6, lon: 139.7 },  // Tokyo
  { lat: 48.8, lon: 2.3 },    // Paris
  { lat: -33.8, lon: 151.2 }, // Sydney
  { lat: 37.5, lon: 127.0 },  // Seoul
  { lat: 19.0, lon: 72.8 },   // Mumbai
];

function latLonToVec3(lat, lon, radius) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

export default function CrisisGlobe({ viralityIndex, crisisLevel, sentimentScore }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const animRef = useRef(null);
  const hotspotMeshesRef = useRef([]);
  const particlesRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const W = mount.clientWidth;
    const H = mount.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1000);
    camera.position.z = 2.8;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);
    sceneRef.current = { scene, camera, renderer };

    // Globe
    const globeGeo = new THREE.SphereGeometry(1, 64, 64);
    const globeMat = new THREE.MeshPhongMaterial({
      color: CRISIS_COLORS[crisisLevel]?.globe ?? 0x1a3a5c,
      emissive: 0x0a0a1a,
      specular: 0x333366,
      shininess: 60,
      wireframe: false,
      transparent: true,
      opacity: 0.92,
    });
    const globe = new THREE.Mesh(globeGeo, globeMat);
    scene.add(globe);

    // Wireframe overlay
    const wireGeo = new THREE.SphereGeometry(1.005, 32, 32);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x1a3a6a,
      wireframe: true,
      transparent: true,
      opacity: 0.15,
    });
    scene.add(new THREE.Mesh(wireGeo, wireMat));

    // Atmosphere glow
    const atmGeo = new THREE.SphereGeometry(1.08, 32, 32);
    const atmMat = new THREE.MeshPhongMaterial({
      color: 0x0044aa,
      transparent: true,
      opacity: 0.08,
      side: THREE.BackSide,
    });
    scene.add(new THREE.Mesh(atmGeo, atmMat));

    // Lighting
    scene.add(new THREE.AmbientLight(0x222244, 0.8));
    const dirLight = new THREE.DirectionalLight(0x8888ff, 1.2);
    dirLight.position.set(5, 3, 5);
    scene.add(dirLight);
    const rimLight = new THREE.DirectionalLight(0x002266, 0.6);
    rimLight.position.set(-5, -2, -3);
    scene.add(rimLight);

    // Crisis hotspots
    const hotspots = [];
    HOTSPOT_POSITIONS.forEach((pos, i) => {
      const active = i < Math.ceil(viralityIndex * HOTSPOT_POSITIONS.length);
      if (!active) return;

      const vec = latLonToVec3(pos.lat, pos.lon, 1.02);
      const color = CRISIS_COLORS[crisisLevel]?.hotspot ?? 0x00d4ff;

      // Ring
      const ringGeo = new THREE.TorusGeometry(0.04, 0.008, 8, 32);
      const ringMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.copy(vec);
      ring.lookAt(0, 0, 0);
      scene.add(ring);

      // Dot
      const dotGeo = new THREE.SphereGeometry(0.012, 8, 8);
      const dotMat = new THREE.MeshBasicMaterial({ color });
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(vec);
      scene.add(dot);

      hotspots.push({ ring, dot, phase: Math.random() * Math.PI * 2 });
    });
    hotspotMeshesRef.current = hotspots;

    // Particle field for virality
    const particleCount = Math.floor(viralityIndex * 200);
    if (particleCount > 10) {
      const particleGeo = new THREE.BufferGeometry();
      const positions = new Float32Array(particleCount * 3);
      for (let i = 0; i < particleCount; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 1.1 + Math.random() * 0.4;
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.cos(phi);
        positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
      }
      particleGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const particleMat = new THREE.PointsMaterial({
        color: CRISIS_COLORS[crisisLevel]?.hotspot ?? 0x00d4ff,
        size: 0.015,
        transparent: true,
        opacity: 0.6,
      });
      const particles = new THREE.Points(particleGeo, particleMat);
      scene.add(particles);
      particlesRef.current = particles;
    }

    // Animation loop
    let t = 0;
    const animate = () => {
      animRef.current = requestAnimationFrame(animate);
      t += 0.005;

      globe.rotation.y += 0.003;
      if (particlesRef.current) particlesRef.current.rotation.y -= 0.001;

      hotspotMeshesRef.current.forEach(({ ring, phase }) => {
        ring.scale.setScalar(1 + 0.2 * Math.sin(t * 2 + phase));
        ring.material.opacity = 0.6 + 0.3 * Math.sin(t * 3 + phase);
      });

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animRef.current);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, [viralityIndex, crisisLevel, sentimentScore]);

  const glowColor = CRISIS_COLORS[crisisLevel]?.glow ?? "#00d4ff";

  return (
    <div className="globe-container" style={{ "--glow": glowColor }}>
      <motion.div
        className="globe-glow-ring"
        animate={{ scale: [1, 1.05, 1], opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{ borderColor: glowColor, boxShadow: `0 0 40px ${glowColor}44` }}
      />
      <div ref={mountRef} className="globe-mount" />
    </div>
  );
}
