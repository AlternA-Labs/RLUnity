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
       
       [SerializeField] private int phase1Steps = 650;      // Evre‑1: salt Y kontrol eskiden 4000
       [SerializeField] private int phase2Steps = 1300;//+650
       [SerializeField] private int phase3Steps = 2600; // Evre‑3 bitişi (kümülatif)14000
       [SerializeField] private float riseSpeedPhase1 = 0.005f; // Y ekseni artış (hızlı)
       [SerializeField] private float riseSpeedPhase2 = 0.005f; // Y ekseni artış (yavaş)
       [SerializeField] private float maxRiseY = 4.5f; 
       [SerializeField] private int maxEpisodeSteps = 500;
    
    
        [Header("Movement Settings")]
        [SerializeField] private float pitchSpeed = 100f;
        
        [SerializeField] private float thrustForce;//thrust force u delta time ile carpabilmek icin episode begine aldim.

        [Header("Penalties / Rewards")]
        [SerializeField] private float pitchPenalty = 0.01f;
        [SerializeField] private Transform sensorRoot;   // Inspector’dan SensorRoot'u sürükle

        [SerializeField] private float movePenalty = 0.05f;  // pitch kullanım cezası
        [SerializeField] private float stepPenalty = 0.005f;  // Zaman cezası (her adım)
        [SerializeField] private float tiltPenalty = 0.02f;   // Yan yatma cezası (her adım)
        [FormerlySerializedAs("AlaboraPenalty")] 
        [SerializeField] private float alaboraPenalty = 1f;  


        [Header("Stability Reward Settings")]
        [SerializeField] private float stableVelocityThreshold = 0.1f;
        [SerializeField] private float stableAngleThreshold = 5f;  // Kaç derecenin altı dik sayılacak
        [SerializeField] private float stableReward = 0.5f; 

        [Header("Approach Reward")]
        [SerializeField] private float approachRewardFactor = 0.1f;  //ekponansiyel ödül katsayısı
        [SerializeField] public float tiltThreshold = 10f;            // Başlangıç ceza eşiği
        [SerializeField] private float recoveryReward = 0.1f;        // Düzeltme ödül katsayısı
        [SerializeField] private float extremeTiltAngle = 80f;        // Aşırı sapma eşiği (örneğin 80 derece)
        [SerializeField] private float penaltyInterval = 1f;
        [Header("Astro Position")]

        
        [SerializeField] private float astroStepFactor = 0.000008f;
    
        // Önceki mesafeyi tutarak yaklaşma/uzaklaşma hesabı
        private float _previousDistanceToAstro = 0f;
        private float _tiltTimeAccumulator = 0f; 
        private float _nextPenaltyThreshold = 1f;
        private GameObject m_LandObject;
        private float counter = 0f;
        private int episodeIndex = 0;
        private int episodeStep = 0; 
        private long stepCount = 0;

// Yardımcı:
        private enum Phase { One, Two, Three }

        private Phase CurrentPhase =>
            episodeIndex < phase1Steps ? Phase.One :
            episodeIndex < phase2Steps ? Phase.Two :
            episodeIndex < phase3Steps ? Phase.Three :
            Phase.Three;

 
        //ReSharper disable Unity.PerformanceAnalysis
        // Loglama için yeni eklenen alanlar
        private string logFilePath;
        private StreamWriter logWriter;
        private bool isLogWriterClosed = true;//LOG TUTSUN MU?
        
        private GameObject  _astroGO;
        private SkinnedMeshRenderer _astroRenderer;
        private BoxCollider _astroCollider;
        private float carpan = 0.5f ;

        // Başlangıçta log dosyasını ayarla
        private void LogMessage(string message)
        {
            try
            {
                if (logWriter != null && !isLogWriterClosed )
                {
                    logWriter.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} - {message}");
                    logWriter.Flush(); // Her yazmada dosyaya kaydedilmesini sağlar
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
            
            BoxCollider   rocketCol = GetComponent<BoxCollider>();          // gövde
            BoxCollider sensorCol = GetComponentInChildren<BoxCollider>(); // kafadaki
            if (rocketCol != null && sensorCol != null)
            {
                Physics.IgnoreCollision(rocketCol, sensorCol, true); 
            }//titereme sorunu olmasın diye sonsor ve roketin rigidbodylerindaki çakışmayı engelledik.
            
            
            Debug.Log("Log dosyası yolu: " + logFilePath);
            // Log dosyasının yolu: PersistentDataPath kullanarak platformdan bağımsız bir yol
            
            logFilePath = Path.Combine(Application.persistentDataPath, "RocketAgent_Log1.txt");
            logWriter = new StreamWriter(logFilePath, true); // true: dosyaya ekleme yapar
            isLogWriterClosed = false;
            LogMessage("[INFO] Uygulama başlatıldı: " + DateTime.Now);
            
            _astroGO = Instantiate(spawnAstro, Vector3.zero, Quaternion.identity);
            _astroGO.tag = "Astro";
            astro   = _astroGO.transform;
            _astroRenderer = _astroGO.GetComponentInChildren<SkinnedMeshRenderer>();
            _astroCollider = _astroGO.GetComponentInChildren<BoxCollider>();
            astroDestroyed = false;
        }
        
// Kodu OnEpisodeBegin sonunda çağır
        private void PlaceAstro()
        {
            if (CurrentPhase == Phase.One)
            {
                // Başlangıç (0,0,0)’a yakın + hızlı Y yükselişi
                float y = Mathf.Min(1.17f + episodeIndex * riseSpeedPhase1, maxRiseY);
                astro.position = new Vector3(0f, y, 0f);
            }
            else if (CurrentPhase == Phase.Two)
            {
                // (0,0,0)’a geri dön, yavaş Y yükselişi
                float y = Mathf.Min(1.17f + (episodeIndex - phase1Steps) * riseSpeedPhase2, maxRiseY);
                astro.position = new Vector3(0f, y, 0f);
            }
            else if (CurrentPhase == Phase.Three) // Phase.Three
            {
                float y = Mathf.Min(1.17f + (episodeIndex - phase2Steps) * carpan * riseSpeedPhase2, maxRiseY);
                // Eski mantığı aynen kullan – X‑Z’de uzaklaş + Y’de hafif yüksel
                float boundary = Math.Min(((episodeIndex - phase2Steps) * astroStepFactor)/2,2f);
                float offsetX  = Random.Range(-boundary, boundary);
                float offsetZ  = Random.Range(-boundary, boundary);


                astro.position = new Vector3(transform.position.x + offsetX, y,
                    transform.position.z + offsetZ);
            }
        }

        public override void OnEpisodeBegin()
        {   // Örnek güvenli blok
            if (episodeIndex > phase3Steps)
            {
                Debug.Log("Eğitim tamamlandı – simülasyon durduruluyor.");

                if (Academy.Instance.IsCommunicatorOn)      // Python varsa kapatma!
                {
                    // Trainer koşuyor → yalnızca Episode’i bitir
                    EndEpisode();            // İstersen burada ödül yaz
                }
                else
                {
                    // Stand‑alone / inference
                    Academy.Instance.EnvironmentStep();
                    Academy.Instance.Dispose();             // Tamamen kapat
                    Application.Quit();
                }
                return;
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
            
            
            // Bölüm (episode) başlangıcı
            thrustForce = 1000f * Time.deltaTime;
            
            //ROCKET  posizyon ayarlama.
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

            // 2) X‑Z’de roketin yakınında
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

            // Yeni mesafeyi kaydet
            _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            // Astro mesafesini kaydet
            if (!astroDestroyed)
            {
                _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            }
            
            LogMessage($"[Action] Episode başladı - Pozisyon: {transform.localPosition}, Hız: {rb.linearVelocity}");
            PlaceAstro();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            /*
            // 1) Astro’ya birim yön vektörü (3 float)
            Vector3 dir = (!astroDestroyed 
                              ? astro.position 
                              : landingSite.position)
                          - transform.position;
            sensor.AddObservation(dir.normalized);
           // Debug.Log($"Astro position: {dir.normalized}");

            // 2) Ham mesafe (1 float)
            sensor.AddObservation(dir.magnitude);

            // 3) Hedefe yönelik hız bileşeni (1 float)
            float fwdSpeed = Vector3.Dot(rb.linearVelocity, dir.normalized);
            sensor.AddObservation(fwdSpeed);

            // 4) Roketin “up” vektörü (3 float)
            sensor.AddObservation(transform.up);

            // 5) Ham hız vektörü (3 float)
            sensor.AddObservation(rb.linearVelocity);
            // 6) Raycast ile duvar mesafeleri (5 yönde)
            
            RaycastHit hit;
            
            float forwardDist = Physics.Raycast(transform.position, transform.forward, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(forwardDist);

            float leftDist = Physics.Raycast(transform.position, -transform.right, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(leftDist);

            float rightDist = Physics.Raycast(transform.position, transform.right, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(rightDist);

            float backDist = Physics.Raycast(transform.position, -transform.forward, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(backDist);
            // UP (yukarı)
            float upDist = Physics.Raycast(transform.position, transform.up, out hit, 15f) ? hit.distance : 15f;
            sensor.AddObservation(upDist);
            //Debug.Log($"forward: {forwardDist}, left: {leftDist}, right: {rightDist}, back: {backDist}, up: {upDist}");
*/
            // 1) Hedef vektörü
            Vector3 dir = (!astroDestroyed ? astro.position : landingSite.position) 
                          - sensorRoot.position;
            sensor.AddObservation(dir.normalized);

            // 2) Mesafe
            sensor.AddObservation(dir.magnitude);

            // 3) İleri hız bileşeni
            float fwdSpeed = Vector3.Dot(rb.linearVelocity, dir.normalized);
            sensor.AddObservation(fwdSpeed);

            // 4) Up vektörü
            sensor.AddObservation(sensorRoot.up);

            // 5) Ham hız
            sensor.AddObservation(rb.linearVelocity);

            // 6) Raycast’ler – sensorRoot’tan at
            RaycastHit hit;
            float forwardDist = Physics.Raycast(sensorRoot.position, sensorRoot.forward, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(forwardDist);
            
            float leftDist = Physics.Raycast(sensorRoot.position, -transform.right, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(leftDist);

            float rightDist = Physics.Raycast(sensorRoot.position, transform.right, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(rightDist);

            float backDist = Physics.Raycast(sensorRoot.position, -transform.forward, out hit, 10f) ? hit.distance : 10f;
            sensor.AddObservation(backDist);
            // UP (yukarı)
            float upDist = Physics.Raycast(sensorRoot.position, transform.up, out hit, 15f) ? hit.distance : 15f;
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
                AddReward(-10f);   // ceza
                EndEpisode();
                return;            // kalan kodu çalıştırma
            }
            
            bool freezePhase = CurrentPhase == Phase.One;

            if (freezePhase)
            {
                // 1) Pozisyon ve rotasyonu kilitle
                rb.constraints = RigidbodyConstraints.FreezePositionX |
                                 RigidbodyConstraints.FreezePositionZ |
                                 RigidbodyConstraints.FreezeRotation;

                // 2) Yatay hız/rotasyonu sıfırla (kayma olmasın)
                rb.linearVelocity  = new Vector3(0f, rb.linearVelocity.y, 0f);
                rb.angularVelocity = Vector3.zero;

                // 3) Sadece Y eksenine itki uygula → ajan “dikey itki”yi öğreniyor
                float thrustOnly = actions.ContinuousActions[2];
                rb.AddForce(thrustOnly * thrustForce * transform.up);

                return;                     // Pitch‑Yaw henüz işlenmedi
            }
            else if (rb.constraints != RigidbodyConstraints.None)
            {
                // Freeze bitti, kısıtlamayı tek seferde kaldır
                rb.constraints = RigidbodyConstraints.None;
            }

//  ─────────────────────────────────────────────────────────────
//  Normal kontrol (X‑Z serbest)
//  ─────────────────────────────────────────────────────────────


            float pitchInputX = actions.ContinuousActions[0];
            float pitchInputZ = actions.ContinuousActions[1];
            float thrustInput = actions.ContinuousActions[2];

            //pitch için abs yapıp yapılan pitch *0.1 ceza olsun. unutma.
            LogMessage($"[Action] PitchX: {pitchInputX}, PitchZ: {pitchInputZ}, Thrust: {thrustInput}");

            // Aksiyon sıfırdan farklıysa ufak ceza
            if (Mathf.Abs(pitchInputX) > 1e-6f)
            {
                AddReward(-pitchPenalty);
                counter-=0.001f;
                LogMessage($"[Reward] PitchX hareket cezası: {-movePenalty}");
                
            }
            
            if (Mathf.Abs(pitchInputZ) > 1e-6f)
            {
                AddReward(-pitchPenalty);
                counter-=0.001f;
                LogMessage($"[Reward] PitchZ hareket cezası: {-movePenalty}");
                
            }
            
            /*
            if (Mathf.Abs(thrustInput) > 1e-6f)
            {
                AddReward(4 * movePenalty);
                counter+=4 * movePenalty;
                LogMessage($"[Reward] Thrust ödülü: {4 * movePenalty}");
            }
            */
 
            // Roketi yönlendir
            transform.Rotate(pitchInputX * pitchSpeed * Time.deltaTime, 0f, 0f);
            transform.Rotate(0f, 0f, pitchInputZ * pitchSpeed * Time.deltaTime);
            rb.AddForce(thrustInput * thrustForce * transform.up);

            // Yan yatma
            float angleFromUp = Vector3.Angle(transform.up, Vector3.up);
            //Debug.Log($"angleFromUp: {angleFromUp}");
        
            LogMessage($"[State] Pozisyon: {transform.localPosition}, Hız: {rb.linearVelocity}, Açı: {angleFromUp}");

            if (angleFromUp > tiltThreshold)
            {       
                // Agent tilt durumunda: zaman sayalım
                _tiltTimeAccumulator += Time.deltaTime;
    
                // Eğer bir ceza periyodu (penaltyInterval) geçtiyse:
                if (_tiltTimeAccumulator >= _nextPenaltyThreshold)
                {
                    // Aşım miktarını hesaplayalım:
                    float currentTiltError = angleFromUp - tiltThreshold;
                    // Temel ceza: Aşım miktarına göre ceza
                    float basePenalty = tiltPenalty * currentTiltError;
                    AddReward(-basePenalty);
                    counter -=basePenalty ;
                    LogMessage($"[Reward] Eğim cezası: {-basePenalty}");
        
                    // Eğer açı extreme değerin üzerinde ise (bağımsız olarak ek ceza):
                    if (angleFromUp >= extremeTiltAngle)
                    {
                        AddReward(-alaboraPenalty);
                        counter -=alaboraPenalty ;
                        LogMessage($"[Reward] Aşırı eğim cezası: {-alaboraPenalty}");
                    }

                    if (angleFromUp >= 89f)
                    {
                        AddReward(-10f);
                        Debug.Log("Taklaya Geldik");
                        EndEpisode();
                    }
        
                    // Sonraki ceza periyodunu ayarla:
                    _nextPenaltyThreshold += penaltyInterval;
                }
            }
            else
            {
                // Agent açı eşik altına düştüyse (yani kendini düzelttiyse)
                if (_tiltTimeAccumulator > penaltyInterval)
                {       
                    // Eğer tilt durumu en az penaltyInterval sürdüyse, fazladan kalan süre için ödül ver
                    //float recoveryTime = _tiltTimeAccumulator - penaltyInterval;
                    //float rew = recoveryReward * recoveryTime;
                    //AddReward(rew);
                    //counter += rew;
                    //Debug.Log($"Kendnini Düzeltme ödülü:{rew}");
                    //LogMessage($"[Reward] Düzeltme ödülü: {rew}");
                }
                // Sayaçları sıfırlayalım:
                _tiltTimeAccumulator = 0f;
                _nextPenaltyThreshold = penaltyInterval;
            }
            
            


            // 2) Tamamen ters dönmüş (örneğin 150 derece üstü) sayıyoruz
            //    Bu durumda büyük ceza ve bölümü bitirmek isteyebilirsiniz
            /*if (angleFromUp > 80f)// bi ara 90 bi ara da 150 idi
        {
            //model cok hizli kaybettigi icin ogrenemiyor 
            AddReward(-AlaboraPenalty);
            //EndEpisode();
            //return; // Bitti, devam etmeye gerek yok
        }
        */
            // Zaman cezası
            //AddReward(-stepPenalty);


            // // Astro'ya yaklaşma ödülü (distance shaping)
            // if (!astroDestroyed)
            // {
            //     float currentDistance = Vector3.Distance(transform.position, astro.position);
            //     float distanceDelta = _previousDistanceToAstro - currentDistance;  // + ise yaklaştık, - ise uzaklaştık
            //     // Debug.Log($"distanceDelta: {distanceDelta}");
            //     AddReward(distanceDelta * 5 * approachRewardFactor);
            //
            //     _previousDistanceToAstro = currentDistance;
            //
            //  
            //     // Roket eylemlerini uyguladıktan sonra:
            //     float speed = rb.linearVelocity.magnitude;
            //     angleFromUp = Vector3.Angle(transform.up, Vector3.up);
            //
            //     if (speed < stableVelocityThreshold && angleFromUp < stableAngleThreshold)
            //     {
            //         //AddReward(5*stableReward); // eski lineer ödül mekanizması.
            //         bool isApproaching = distanceDelta > 0;
            //         float sign = isApproaching ? 1f : -1f;
            //         float absDelta = Mathf.Abs(distanceDelta);
            //
            //         float scalingFactor = 0.5f;
            //         float baseReward = Mathf.Exp(absDelta * scalingFactor) - 1f;
            //         float expReward = baseReward * approachRewardFactor * sign;
            //
            //         AddReward(expReward);
            //         Debug.Log($"Approach: {isApproaching} DistanceDelta: {distanceDelta}, ExpReward: {expReward} Previous: {_previousDistanceToAstro}, Current: {currentDistance}");
            //
            //         // exponansiyel yaklasma odulu sonu
            //     }
            //
            // }

            
 /*           if (Vector3.Distance(transform.position, astro.position) < 1f && !astroDestroyed)//manuel carpisma
            {
                    OnTriggerEnter(astroCollider.GetComponent<Collider>());
            }
   */         
            
            if (m_LandObject != null)
            {
                // Kendi konumunuz ile "land" objesinin konumunu al ve mesafeyi hesapla
                float distance = Vector3.Distance(transform.position, m_LandObject.transform.position);
                if (distance < 1.0f)
                {
                    //Debug.Log("eşşeklik cezası");
                    AddReward(-0.05f);
                    counter -= 0.05f;
                    LogMessage("[Reward] Landing alanına yakınlık cezası: -0.05");
                }
            }
            else
            {
                Debug.LogWarning("Land etiketli obje bulunamadı!");
            }

            /* ----- YENİ: hizalanma + ileri hız ödülü ----- */
            if (!astroDestroyed)
                
            {
                float speed = rb.linearVelocity.magnitude;
                angleFromUp = Vector3.Angle(transform.up, Vector3.up);
                Vector3 dir       = (astro.position - transform.position).normalized;
                float   align     = Mathf.Max(Vector3.Dot(transform.up, dir), 0f);       // negatifleri yok say
                float   fwdSpeed  = Mathf.Max(Vector3.Dot(rb.linearVelocity, dir), 0f);        // sadece ileri yöndeki hız

                // normalize edilmiş hız /5f, aynı ağırlıkta küçük katsayılarla
                LogMessage($"[*] Step {stepCount}  CumReward: {GetCumulativeReward():F3} , -KONUM BAŞLANGICI-");
                //AddReward(0.01f * align + 0.01f * (fwdSpeed / 5f));
                
                AddReward(0.002f * align);           // 5× küçülttük
                AddReward(0.0004f * fwdSpeed);       // 5× küçülttük


                LogMessage($"[Reward] TargetSeeking: align={align:F2}, speed={fwdSpeed:F2}");
            }
            
            LogMessage($"[*] Step {stepCount}  CumReward: {GetCumulativeReward():F3} -KONUM SONU-");

            
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Manuel kontrol
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
            counter += 10f;//eski hali =20, v1.2 güncellemesi
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

            /*if (other.TryGetComponent<AstroScript>(out AstroScript half))
            {
                AddReward(10f);
                counter += 10f;//eski hali =20, v1.2 güncellemesi
 //               astroRenderer.gameObject.GetComponent<SkinnedMeshRenderer>().enabled = false;
  //              astroCollider.gameObject.GetComponent<BoxCollider>().enabled = false;
                 _astroRenderer.enabled = false;
                 _astroCollider.enabled  = false;
                LogMessage("[Reward] Astro çarpması ödülü: 10");
                
                astroDestroyed = true;
                Debug.Log("Astro Collision, rewarded");
                EndEpisode();//v1.2 eklemesi
            }
            
            */
        
            // Duvara çarparsa -1 ve bölüm sonu
            if (other.TryGetComponent<Wall>(out Wall fail))
            {
                //eski degerler reward : -1
                SetReward(-10f);
                LogMessage("[Reward] Duvar çarpması cezası: -20");
                EndEpisode();   
            }

            
        }
        
       
      
        // landzone'a belirlenen şartlarla inerse +0.5 ve bölüm sonu
        
        void OnCollisionEnter(Collision col)//triggerda bozuk calistigi icin OnCollisiona gecis yapildi
    {
        /*
        if(col.gameObject.CompareTag("land")){
            Debug.Log("touchdown");
            float rotx = transform.localRotation.eulerAngles.x;
            float rotz = transform.localRotation.eulerAngles.z;
            //Debug.Log($"astroDestroyed: {astroDestroyed}");
            //Debug.Log("ekinto");
            
            if (astroDestroyed && (rotx >= -2.5f && rotx <= 2.5f || rotx >= 357.5f)//aci kontrolleri guncellendi
                               && (rotz >= -2.5f && rotz <= 2.5f || rotz >= 357.5f))
            {
                AddReward(20f);
                counter += 20f;
                LogMessage("[Reward] Başarılı iniş ödülü: 20");
                Debug.Log("Success");
                counter = 0f;
                EndEpisode();
            }
        }

    */}

    void Update()
    {
        /*
        float previousDistanceForLog = _previousDistanceToAstro;
        float currentDistance = Vector3.Distance(transform.position, astro.position);
        float distanceDelta = _previousDistanceToAstro - currentDistance;
        bool isApproaching = distanceDelta > 0;
        float Reward = distanceDelta ;
        AddReward(Reward);
        counter += Reward;
        LogMessage($"[Reward] Yaklaşma/uzaklaşma ödülü: {Reward}, Mesafe: {currentDistance}");
        LogMessage($"[*] Current Reward: {GetCumulativeReward()}");
        Debug.Log($"Approach: {isApproaching} Counter: {counter}, Distance: {currentDistance}, Delta: {distanceDelta}");
        if (currentDistance>15f)
        {
            counter = 0f;
            Debug.Log($"uzaklaşma cezası.");
            AddReward(-10f);
            LogMessage("[Action] Uzaklaşma cezası, episode sonu");
            EndEpisode();
        }
        
        //Debug.Log($"Approach: {isApproaching} DistanceDelta: {distanceDelta}, ExpReward: {Reward} Previous: {previousDistanceForLog}, Current: {currentDistance}");

        _previousDistanceToAstro = currentDistance;
        */
    }
    // Uygulama kapanırken log dosyasını kapat
    private void OnApplicationQuit()
    {
        if (!isLogWriterClosed && logWriter != null)
        {
            LogMessage("[INFO] Uygulama kapatıldı");
            logWriter.Close();
            isLogWriterClosed = true;
        }
    }

    // Hata durumunda loglama ve dosyayı kapatma
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
