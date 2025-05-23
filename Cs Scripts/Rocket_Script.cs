using System;
using System.Diagnostics;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Serialization;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace RLUnity.Cs_Scripts
{
    public class RocketAgent : Agent
    {
        [Header("References")]
        [SerializeField] private Transform astro;
        [SerializeField] private Transform landingSite;
        [SerializeField] private Rigidbody rb;
        [SerializeField] private GameObject spawnAstro;
        [FormerlySerializedAs("AstroDestroyed")] 
        [SerializeField] private bool astroDestroyed;

       [Header("Step Settings")]
       
       [SerializeField] private int phase1Steps = 650;     
       [SerializeField] private int phase2Steps = 800;
       [SerializeField] private int phase3Steps = 1450; 
       [SerializeField] private int phase4Steps = 1850;
       [SerializeField] private int phase5Steps = 2850;
       [SerializeField] private float riseSpeedPhase1 = 0.005f; 
       [SerializeField] private float riseSpeedPhase2 = 0.005f;
       [SerializeField] private float maxRiseY = 4.0f; 
       [SerializeField] private int maxEpisodeSteps = 500;
    
    
        [Header("Movement Settings")]
        [SerializeField] private float pitchSpeed = 100f;
        
        [SerializeField] private float thrustForce;

        [Header("Penalties / Rewards")]
        [SerializeField] private float pitchPenalty = 0.05f;
        [SerializeField] private Transform sensorRoot;

        //[SerializeField] private float movePenalty = 0.05f;
        [SerializeField] private float stepPenalty = 0.003f;  
        //[SerializeField] private float tiltPenalty = 0.02f;  

        //[Header("Stability Reward Settings")]
        //[SerializeField] private float stableVelocityThreshold = 0.1f;
        //[SerializeField] private float stableAngleThreshold = 5f;
        //[SerializeField] private float stableReward = 0.5f; 

        [Header("Approach Reward")]
       // [SerializeField] private float approachRewardFactor = 0.1f;
        [SerializeField] public float tiltThreshold = 10f;           
        //[SerializeField] private float recoveryReward = 0.1f;        
        [SerializeField] private float penaltyInterval = 1f;
        [Header("Astro Position")]

        
        [SerializeField] private float astroStepFactor = 0.000008f;
        
        private float _previousDistanceToAstro = 0f;
        private float _tiltTimeAccumulator = 0f; 
        private float _nextPenaltyThreshold = 1f;
        bool trainingFinished = false;
        private GameObject m_LandObject;
        private float counter = 0f;
        private int episodeIndex = 0;
        private int episodeStep = 0; 
        private long stepCount = 0;
        private bool isTraining ;


        private enum Phase { One, Two, Three, Four, Five }

        private Phase CurrentPhase =>
            episodeIndex < phase1Steps ? Phase.One :
            episodeIndex < phase2Steps ? Phase.Two :
            episodeIndex < phase3Steps ? Phase.Three :
            episodeIndex < phase4Steps ? Phase.Four :
            episodeIndex < phase5Steps ? Phase.Five :
            Phase.Five;

 
        //ReSharper disable Unity.PerformanceAnalysis
        private string logFilePath;
        private StreamWriter logWriter;
        private bool isLogWriterClosed = true;
        
        private GameObject  _astroGO;
        private SkinnedMeshRenderer _astroRenderer;
        private BoxCollider _astroCollider;
        private float carpan = 0.5f ;

        
        private void LogMessage(string message)
        {
            try
            {
                if (logWriter != null && !isLogWriterClosed )
                {
                    logWriter.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} - {message}");
                    logWriter.Flush(); 
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Log yazma hatası: {e.Message}");
            }
        }
        private void Awake()
        {
            
            if (sensorRoot == null) sensorRoot = transform;
            
            BoxCollider   rocketCol = GetComponent<BoxCollider>();          
            BoxCollider sensorCol = GetComponentInChildren<BoxCollider>();
            if (rocketCol != null && sensorCol != null)
            {
                Physics.IgnoreCollision(rocketCol, sensorCol, true); 
            }
            
            
            Debug.Log("Log dosyası yolu: " + logFilePath);
            
            
            logFilePath = Path.Combine(Application.persistentDataPath, "RocketAgent_Log1.txt");
            logWriter = new StreamWriter(logFilePath, true);
            isLogWriterClosed = false;
            LogMessage("[INFO] Uygulama başlatıldı: " + DateTime.Now);
            
            _astroGO = Instantiate(spawnAstro, Vector3.zero, Quaternion.identity);
            _astroGO.tag = "Astro";
            astro   = _astroGO.transform;
            _astroRenderer = _astroGO.GetComponentInChildren<SkinnedMeshRenderer>();
            _astroCollider = _astroGO.GetComponentInChildren<BoxCollider>();
            astroDestroyed = false;
        }

        public override void Initialize()
        {
            var envParams = Academy.Instance.EnvironmentParameters;
            float flag = envParams.GetWithDefault("is_training", 1f);
            isTraining = flag > 0.5f;
            Debug.Log(isTraining);
        }
        

        private void PlaceAstro()
        {
            if (CurrentPhase == Phase.One && isTraining)
            {
                
                float y = Mathf.Min(1.17f + episodeIndex * riseSpeedPhase1, maxRiseY);
                astro.position = new Vector3(0f, y, 0f);
            }
            else if (CurrentPhase == Phase.Two&& isTraining)
            {
                
                float y = Mathf.Min(1.17f + (episodeIndex - phase1Steps) * riseSpeedPhase2, maxRiseY);
                astro.position = new Vector3(0f, y, 0f);
            }
            else if (CurrentPhase == Phase.Three&& isTraining)
            {
                
                float y = Mathf.Min(1.17f + (episodeIndex - phase2Steps) * riseSpeedPhase2, maxRiseY);
                astro.position = new Vector3(0f, y, 0f);
            }
            else if (CurrentPhase == Phase.Four && isTraining) // Phase.Three
            {
                float y = Mathf.Min(1.17f + (episodeIndex - phase3Steps) * carpan * riseSpeedPhase2, maxRiseY);
                float height    = y - 0.877f;
                
                float maxRadius = Mathf.Tan(30f * Mathf.Deg2Rad) * height;
                Vector2 rnd     = Random.insideUnitCircle * maxRadius;
                float offsetX   = rnd.x;
                float offsetZ   = rnd.y;
                /*
                float boundary2 = (y - 0.877f)/2f;
                
                float boundary = Math.Min(((episodeIndex - phase2Steps) * astroStepFactor)/2,boundary2);
                float offsetX  = Random.Range(-boundary, boundary);
                float offsetZ  = Random.Range(-boundary, boundary);
*/

                astro.position = new Vector3(transform.position.x + offsetX, y,
                    transform.position.z + offsetZ);
            }
            else if (CurrentPhase == Phase.Five && isTraining) // Phase.Three
            {
                float y = Mathf.Min(1.17f + (episodeIndex - phase4Steps) * carpan * riseSpeedPhase2, maxRiseY);
                float height    = y - 0.877f;
                
                float maxRadius = Mathf.Tan(30f * Mathf.Deg2Rad) * height;
                Vector2 rnd     = Random.insideUnitCircle * maxRadius;
                float offsetX   = rnd.x;
                float offsetZ   = rnd.y;
                /*
                float boundary2 = (y - 0.877f)/2f;
                
                float boundary = Math.Min(((episodeIndex - phase2Steps) * astroStepFactor)/2,boundary2);
                float offsetX  = Random.Range(-boundary, boundary);
                float offsetZ  = Random.Range(-boundary, boundary);
*/

                astro.position = new Vector3(transform.position.x + offsetX, y,
                    transform.position.z + offsetZ);
            }
            else if (!isTraining)
            {
                Debug.Log("TEST");
                float y = Random.Range(1.30f, 3.5f);
                float height    = y - 0.877f;
                
                float maxRadius = Mathf.Tan(30f * Mathf.Deg2Rad) * height;
                Vector2 rnd     = Random.insideUnitCircle * maxRadius;
                float offsetX   = rnd.x;
                float offsetZ   = rnd.y;
                
                astro.position = new Vector3(transform.position.x + offsetX, y,
                    transform.position.z + offsetZ);
            }
        }

        public override void OnEpisodeBegin()
        {  
            if (episodeIndex > phase5Steps)
            {
                Debug.Log("Eğitim tamamlandı – simülasyon durduruluyor.");
                
                
                Academy.Instance.EnvironmentStep(); 
                Academy.Instance.Dispose();           
#if UNITY_EDITOR
                EditorApplication.isPlaying = false;
#else
                Application.Quit();
#endif

           
            }

            Debug.Log($"Episode: {episodeIndex}, Phase: {CurrentPhase}");
            episodeStep = 0;  
            SetReward(0f);
            episodeIndex++;
            LogMessage("");
            LogMessage($"--------- EPISODE {episodeIndex} START ---------");   
            counter = 0f;
            
            m_LandObject= GameObject.FindGameObjectWithTag("land");
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 70;
            
            
            
            thrustForce = 1000f * Time.deltaTime;
            
            
            /*transform.localPosition = new Vector3(
                Random.Range(-1f, 1f),
                0.01f,
                Random.Range(-1f, 1f));
            */
            
            
            transform.localPosition= new Vector3(0f,0.01f,0f);
            
            transform.eulerAngles = new Vector3(0f, 180f, 0f);
            rb.Sleep();
            
            /*
            float newAstroY;
            if (stepCount < 6000)
            {

                newAstroY = Mathf.Min(
                    1.3f + stepCount * astroStepFactor,   
                    4.50f                              
                );
            }
            else
            {
                newAstroY = 2.5f;
            }

            
            float offsetX, offsetZ;
            if (stepCount < 6000)
            {
                offsetX = Random.Range(0f, 0f);//-+2 idi
                offsetZ = Random.Range(0f, 0f);//-+2 idi
            }
            else
            {
                float boundary = Math.Min(((stepCount - 6000f) * astroStepFactor)/2,2f);
                boundary  *= 0.4f;
                Debug.Log($"Hesaplama değeri * : {boundary}");
                offsetX = Random.Range(-boundary, boundary);//-+2 idi
                offsetZ = Random.Range(-boundary, boundary);//-+2 idi
            }


            astro.position = new Vector3(
                transform.position.x + offsetX,
                newAstroY,
                transform.position.z + offsetZ
            );
            */
            
            
            Debug.Log($"stepCount: {stepCount}");
            Debug.Log($"astroY: {astro.position.y}");
            _astroRenderer.enabled = true;
            _astroCollider.enabled  = true;
            astroDestroyed          = false;
            _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            
            _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            
            if (!astroDestroyed)
            {
                _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            }
            
            LogMessage($"[Action] Episode başladı - Pozisyon: {transform.localPosition}, Hız: {rb.linearVelocity}");
            PlaceAstro();
        }

        public override void CollectObservations(VectorSensor sensor)
{
    // Compute direction vector: towards the asteroid if it exists, otherwise towards the landing site
    Vector3 dir = (!astroDestroyed ? astro.position : landingSite.position)
                  - sensorRoot.position;
    // 1) Normalized direction vector: indicates the direction to the target
    sensor.AddObservation(dir.normalized);
    
    // 2) Distance to target: the magnitude of the direction vector
    sensor.AddObservation(dir.magnitude);
    
    // 3) Forward speed component: projection of the agent’s velocity onto the target direction
    float fwdSpeed = Vector3.Dot(rb.linearVelocity, dir.normalized);
    sensor.AddObservation(fwdSpeed);
    
    // 4) Agent’s up vector: world-space upward orientation of the sensor root
    sensor.AddObservation(sensorRoot.up);
    
    // 5) Agent’s velocity vector: full 3D linear velocity (x, y, z)
    sensor.AddObservation(rb.linearVelocity);
    
    // Raycast to detect obstacles and measure distances
    RaycastHit hit;
    
    // 6) Distance in front (max 10): distance to obstacle ahead or 10 if none
    float forwardDist = Physics.Raycast(sensorRoot.position, sensorRoot.forward, out hit, 10f)
                        ? hit.distance : 10f;
    sensor.AddObservation(forwardDist);
    
    // 7) Distance to the left (max 10): distance to obstacle to the left or 10 if none
    float leftDist = Physics.Raycast(sensorRoot.position, -transform.right, out hit, 10f)
                     ? hit.distance : 10f;
    sensor.AddObservation(leftDist);

    // 8) Distance to the right (max 10): distance to obstacle to the right or 10 if none
    float rightDist = Physics.Raycast(sensorRoot.position, transform.right, out hit, 10f)
                      ? hit.distance : 10f;
    sensor.AddObservation(rightDist);

    // 9) Distance behind (max 10): distance to obstacle behind or 10 if none
    float backDist = Physics.Raycast(sensorRoot.position, -transform.forward, out hit, 10f)
                     ? hit.distance : 10f;
    sensor.AddObservation(backDist);
    
    // 10) Distance above (max 15): distance to obstacle above or 15 if none
    float upDist = Physics.Raycast(sensorRoot.position, transform.up, out hit, 15f)
                   ? hit.distance : 15f;
    sensor.AddObservation(upDist);
}




        // ReSharper disable Unity.PerformanceAnalysis
        public override void OnActionReceived(ActionBuffers actions)
        
        {
            AddReward(-stepPenalty); 
            
            stepCount++;
            episodeStep++;
            if (episodeStep > maxEpisodeSteps)
            {
                Debug.Log($"[Episode: {episodeIndex}] MAX STEP Reward");
                AddReward(-10f);  
                EndEpisode();
                return;         
            }
            
            bool freezePhase = CurrentPhase == Phase.One;

            if (freezePhase && isTraining)
            {
                rb.constraints = RigidbodyConstraints.FreezePositionX |
                                 RigidbodyConstraints.FreezePositionZ |
                                 RigidbodyConstraints.FreezeRotation;

                rb.linearVelocity  = new Vector3(0f, rb.linearVelocity.y, 0f);
                rb.angularVelocity = Vector3.zero;


                float thrustOnly = actions.ContinuousActions[2];
                rb.AddForce(thrustOnly * thrustForce * transform.up);

                return;                  
            }
            else if (rb.constraints != RigidbodyConstraints.None)
            {

                rb.constraints = RigidbodyConstraints.None;
            }

            
            float pitchInputX = actions.ContinuousActions[0];
            float pitchInputZ = actions.ContinuousActions[1];
            float thrustInput = actions.ContinuousActions[2];
            
            LogMessage($"[Action] PitchX: {pitchInputX}, PitchZ: {pitchInputZ}, Thrust: {thrustInput}");

            if (Mathf.Abs(pitchInputX) > 1e-6f)
            {
                AddReward(-pitchPenalty);
                counter-=0.001f;
                LogMessage($"[Reward] PitchX hareket cezası: {-pitchPenalty}");
                
            }
            
            if (Mathf.Abs(pitchInputZ) > 1e-6f)
            {
                AddReward(-pitchPenalty);
                counter-=0.001f;
                LogMessage($"[Reward] PitchZ hareket cezası: {-pitchPenalty}");
                
            }

            transform.Rotate(pitchInputX * pitchSpeed * Time.deltaTime, 0f, 0f);
            transform.Rotate(0f, 0f, pitchInputZ * pitchSpeed * Time.deltaTime);
            rb.AddForce(thrustInput * thrustForce * transform.up);


            float angleFromUp = Vector3.Angle(transform.up, Vector3.up);
            //Debug.Log($"angleFromUp: {angleFromUp}");
        
            LogMessage($"[State] Pozisyon: {transform.localPosition}, Hız: {rb.linearVelocity}, Açı: {angleFromUp}");

            if (angleFromUp > tiltThreshold)
            {       

                _tiltTimeAccumulator += Time.deltaTime;
    
                
                if (_tiltTimeAccumulator >= _nextPenaltyThreshold)
                {
                    float currentTiltError = angleFromUp - tiltThreshold;
                    //float basePenalty = tiltPenalty * currentTiltError;
                    //AddReward(-basePenalty);
                    //counter -=basePenalty ;

                    if (angleFromUp >= 80f)
                    {

                        counter -=10f ;
                        LogMessage($"[Reward] Extreme Tilt Reward: -10");
                        AddReward(-10f);
                        EndEpisode();
                    }
                    
                    _nextPenaltyThreshold += penaltyInterval;
                }
            }
            else
            {
                _tiltTimeAccumulator = 0f;
                _nextPenaltyThreshold = penaltyInterval;
            }
   
            
            if (m_LandObject != null)
            {
                float distance = Vector3.Distance(transform.position, m_LandObject.transform.position);
                if (distance < 1.0f)
                {
                    AddReward(-0.05f);
                    counter -= 0.05f;
                }
            }
            else
            {
                Debug.LogWarning("Land etiketli obje bulunamadı!");
            }
            
            if (!astroDestroyed)
            {
                float speed = rb.linearVelocity.magnitude;
                angleFromUp = Vector3.Angle(transform.up, Vector3.up);
                Vector3 dir       = (astro.position - transform.position).normalized;
                float   align     = Mathf.Max(Vector3.Dot(transform.up, dir), 0f);     
                float   fwdSpeed  = Mathf.Max(Vector3.Dot(rb.linearVelocity, dir), 0f);        
                

                
                AddReward(0.002f * align);          
                AddReward(0.0004f * fwdSpeed);      


            }
            
            
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
  
            ActionSegment<float> contActions = actionsOut.ContinuousActions;
        
            float pitchx = 0f;
            if (Input.GetKey(KeyCode.S)) pitchx = 1f;
            else if (Input.GetKey(KeyCode.W)) pitchx = -1f;
        
            float pitchz = 0f;
            if (Input.GetKey(KeyCode.D)) pitchz = 1f;
            else if (Input.GetKey(KeyCode.A)) pitchz = -1f;
        
            float thrust = 0f;
            if (Input.GetKey(KeyCode.Space)) thrust = 1f;

            contActions[0] = pitchx;
            contActions[1] = pitchz;
            contActions[2] = thrust;
        }
        
        public void OnAstroHit()
        {
            if (astroDestroyed) return;

            AddReward(10f);
            counter += 10f;
            //astroRenderer.gameObject.GetComponent<SkinnedMeshRenderer>().enabled = false;
            // astroCollider.gameObject.GetComponent<BoxCollider>().enabled = false;
            _astroRenderer.enabled = false;
            _astroCollider.enabled  = false;
            LogMessage("[Reward] Astro çarpması ödülü: 10");
                
            astroDestroyed = true;
            Debug.Log("Astro Collision, rewarded");
            EndEpisode();
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.TryGetComponent<Wall>(out Wall fail))
            {
                SetReward(-10f);
                LogMessage("[Reward] Duvar çarpması cezası: -20");
                EndEpisode();   
            }
            
        }

    private void OnApplicationQuit()
    {
        if (!isLogWriterClosed && logWriter != null)
        {
            LogMessage("[INFO] Uygulama kapatıldı");
            logWriter.Close();
            isLogWriterClosed = true;
        }
    }
    
    private void OnDestroy()
    {
        if (!isLogWriterClosed && logWriter != null)
        {
            LogMessage("[ERROR] Beklenmedik kapatma");
            logWriter.Close();
            isLogWriterClosed = true;
        }
    }

    }
}
