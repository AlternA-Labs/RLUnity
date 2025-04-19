using System;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Serialization;
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
        [SerializeField] private GameObject newAstro;
        [SerializeField] private GameObject astroRenderer;
        [SerializeField] private GameObject astroCollider;
        [FormerlySerializedAs("AstroDestroyed")] 
        [SerializeField] private bool astroDestroyed;
    
    
        [Header("Movement Settings")]
        [SerializeField] private float pitchSpeed = 100f;
        
        [SerializeField] private float thrustForce;//thrust force u delta time ile carpabilmek icin episode begine aldim.

        [Header("Penalties / Rewards")]
        [SerializeField] private float movePenalty = 0.05f;  // pitch kullanım cezası
        [SerializeField] private float stepPenalty = 0.0001f;  // Zaman cezası (her adım)
        [SerializeField] private float tiltPenalty = 0.02f;   // Yan yatma cezası (her adım)
        [FormerlySerializedAs("AlaboraPenalty")] 
        [SerializeField] private float alaboraPenalty = 1f;    // Ters dönünce (tam alabora) verilecek ceza

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
        [SerializeField] private float astroBaseHeight = 1f;      // ilk bölümde yükseklik
        [SerializeField] private float astroStepFactor = 0.0002f;// belirli bir açı için izin verilen süre
    
        // Önceki mesafeyi tutarak yaklaşma/uzaklaşma hesabı
        private float _previousDistanceToAstro = 0f;
        private float _tiltTimeAccumulator = 0f; 
        private float _nextPenaltyThreshold = 1f;
        private GameObject m_LandObject;
        private float counter = 0f;
        private int episodeIndex = 0;
        private long stepCount = 0;


        //ReSharper disable Unity.PerformanceAnalysis
        // Loglama için yeni eklenen alanlar
        private string logFilePath;
        private StreamWriter logWriter;
        private bool isLogWriterClosed = false;

        // Başlangıçta log dosyasını ayarla
        private void LogMessage(string message)
        {
            try
            {
                if (logWriter != null && !isLogWriterClosed)
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
            Debug.Log("Log dosyası yolu: " + logFilePath);
            // Log dosyasının yolu: PersistentDataPath kullanarak platformdan bağımsız bir yol
            logFilePath = Path.Combine(Application.persistentDataPath, "RocketAgent_Log1.txt");
            logWriter = new StreamWriter(logFilePath, true); // true: dosyaya ekleme yapar
            isLogWriterClosed = false;
            LogMessage("[INFO] Uygulama başlatıldı: " + DateTime.Now);
        }

        public override void OnEpisodeBegin()
        {
            episodeIndex++;
            LogMessage("");
            LogMessage($"--------- EPISODE {episodeIndex} START ---------");   // ← ekle
            counter = 0f;
            //sürtünme edkledim !!!!!!
            rb.linearDamping = 1.5f;
            rb.angularDamping = 1f;
            //sonradan silebiliriz 

            m_LandObject= GameObject.FindGameObjectWithTag("land");
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 70;
            // Bölüm (episode) başlangıcı
            thrustForce = 1000f * Time.deltaTime;
            transform.localPosition = new Vector3(0, 1f, 0);
            
            
            astroDestroyed = false;
            // 1) Yeni astro yüksekliği:  y = 0.75  →  4.5  (saturasyonlu)
            float newAstroY = Mathf.Min(
                0.75f + stepCount * astroStepFactor,   // lineer artış
                4.50f                                 // üst sınır
            );

            // 2) X‑Z’de roketin yakınında  (±2 m)
            float offsetX = Random.Range(-2f, 2f);
            float offsetZ = Random.Range(-2f, 2f);

            astro.localPosition = new Vector3(
                transform.localPosition.x + offsetX,
                newAstroY,
                transform.localPosition.z + offsetZ
            );

            // Yeni mesafeyi kaydet
            _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            astroRenderer.gameObject.GetComponent<SkinnedMeshRenderer>().enabled = true;//tekrar gorunur yap
            astroCollider.gameObject.GetComponent<BoxCollider>().enabled = true;//collideri ac (yuksek ihitmalle silincek)

        
            /*if (GameObject.FindWithTag("Astro") == null)
            {
                newAstro = Instantiate(spawnAstro, new Vector3(-0.31f, 4.44f, 0.12f), Quaternion.identity);
                astro = newAstro.transform;
                astroDestroyed = false;
            }
        */
            transform.eulerAngles = new Vector3(0f, 180f, 0f);
            rb.linearVelocity = Vector3.zero;

            // İsterseniz rastgele başlangıç yapabilirsiniz (yorum satırını açın):
        
            transform.localPosition = new Vector3(Random.Range(-1f, 1f),
                0.01f,
                Random.Range(-1f, 1f));
            transform.eulerAngles = new Vector3(0f, 180f, 0f);
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.Sleep();
        
            //x ve z ekseninde hareketi devredisi birak
            // rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        
            // target.localPosition = new Vector3(Random.Range(-4f, 4f),
            // Random.Range(0.5f, 4f),
            //Random.Range(-4f, 4f));
            /*
        astro.localPosition = new Vector3(Random.Range(-4f, 4f),
                                          Random.Range(0.5f, 4f),
                                          Random.Range(-4f, 4f));
        */

            // Astro mesafesini kaydet
            if (!astroDestroyed)
            {
                //_previousDistanceToAstro = Vector3.Distance(transform.localPosition, astro.localPosition);
                _previousDistanceToAstro = Vector3.Distance(transform.position, astro.position);

            }
        
            // =========================
            // STABILITY REWARD KODU
            // =========================
            // 1) Yatay hız çok düşük mü?
            float speed = rb.linearVelocity.magnitude;
            // 2) Dünya yukarısıyla açı
            float angleFromUp = Vector3.Angle(transform.up, Vector3.up);

            // Eğer roket çok devinimsiz ve dikey duruyorsa, ek ödül
            if (speed < stableVelocityThreshold && angleFromUp < stableAngleThreshold)
            {
                AddReward(stableReward);
                counter+=stableReward;
                LogMessage($"[Reward] Stabilite ödülü: {stableReward}");
            }
            LogMessage($"[Action] Episode başladı - Pozisyon: {transform.localPosition}, Hız: {rb.linearVelocity}");
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // 1) Astro’ya birim yön vektörü (3 float)
            Vector3 dir = (!astroDestroyed 
                              ? astro.position 
                              : landingSite.position)
                          - transform.position;
            sensor.AddObservation(dir.normalized);

            // 2) Ham mesafe (1 float)
            sensor.AddObservation(dir.magnitude);

            // 3) Hedefe yönelik hız bileşeni (1 float)
            float fwdSpeed = Vector3.Dot(rb.velocity, dir.normalized);
            sensor.AddObservation(fwdSpeed);

            // 4) Roketin “up” vektörü (3 float)
            sensor.AddObservation(transform.up);

            // 5) Ham hız vektörü (3 float)
            sensor.AddObservation(rb.velocity);
        }


        // ReSharper disable Unity.PerformanceAnalysis
        public override void OnActionReceived(ActionBuffers actions)
        {
            stepCount++;
            float pitchInputX = actions.ContinuousActions[0];
            float pitchInputZ = actions.ContinuousActions[1];
            float thrustInput = actions.ContinuousActions[2];
            //pitch için abs yapıp yapılan pitch *0.1 ceza olsun. unutma.
            LogMessage($"[Action] PitchX: {pitchInputX}, PitchZ: {pitchInputZ}, Thrust: {thrustInput}");

            // Aksiyon sıfırdan farklıysa ufak ceza
            if (Mathf.Abs(pitchInputX) > 1e-6f)
            {
                AddReward(-movePenalty);
                counter-=movePenalty;
                LogMessage($"[Reward] PitchX hareket cezası: {-movePenalty}");
                
            }
            
            if (Mathf.Abs(pitchInputZ) > 1e-6f)
            {
                AddReward(-movePenalty);
                counter-=movePenalty;
                LogMessage($"[Reward] PitchZ hareket cezası: {-movePenalty}");
                
            }
            // thrust için de ceza eklemek isterseniz yorumu açın:
            if (Mathf.Abs(thrustInput) > 1e-6f)
            {
                AddReward(4 * movePenalty);
                counter+=4 * movePenalty;
                LogMessage($"[Reward] Thrust ödülü: {4 * movePenalty}");
            }
 
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
                    float recoveryTime = _tiltTimeAccumulator - penaltyInterval;
                    float rew = recoveryReward * recoveryTime;
                    AddReward(rew);
                    counter += rew;
                    LogMessage($"[Reward] Düzeltme ödülü: {rew}");
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

            
            if (Vector3.Distance(transform.position, astro.position) < 1f && !astroDestroyed)//manuel carpisma
            {
                    OnTriggerEnter(astroCollider.GetComponent<Collider>());
            }
            
            
            if (m_LandObject != null)
            {
                // Kendi konumunuz ile "land" objesinin konumunu al ve mesafeyi hesapla
                float distance = Vector3.Distance(transform.position, m_LandObject.transform.position);
                if (distance < 1.5f)
                {
                    Debug.Log("eşşeklik cezası");
                    AddReward(-0.05f);
                    counter -= 0.05f;
                    LogMessage("[Reward] Landing alanına yakınlık cezası: -0.05");
                }
            }
            else
            {
                Debug.LogWarning("Land etiketli obje bulunamadı!");
            }
            if (!astroDestroyed)
            {
                // Astro'ya yaklaşma ödülü (distance shaping)


                // Geçici değişken ile eski mesafeyi saklayalım
                //float previousDistanceForLog = _previousDistanceToAstro;

                // Ödül hesaplaması
                //AddReward(distanceDelta * 5 * approachRewardFactor);

                // Roket eylemlerini uyguladıktan sonra:
                float speed = rb.linearVelocity.magnitude;
                angleFromUp = Vector3.Angle(transform.up, Vector3.up);

                //bool isApproaching = distanceDelta > 0;
                //float sign = isApproaching ? 1f : -1f;
                //float absDelta = Mathf.Abs(distanceDelta);

                //float scalingFactor = 1f;
                //float baseReward = Mathf.Exp(absDelta * scalingFactor) - 1f;
                //float expReward = baseReward * approachRewardFactor * sign;


                
                

                // Eski mesafeyi güncelle
                //_previousDistanceToAstro = currentDistance;
            }
            /* ----- YENİ: hizalanma + ileri hız ödülü ----- */
            if (!astroDestroyed)
            {
                Vector3 dir      = (astro.position - transform.position).normalized;
                float   align    = Vector3.Dot(transform.up, dir);          // 1 → hedefe bakıyor
                float   fwdSpeed = Vector3.Dot(rb.linearVelocity, dir);     // hedefe doğru hız

                AddReward(0.01f * align);
                AddReward(0.002f * fwdSpeed);
                
                //LogMessage($"[*] StepCumReward: {GetCumulativeReward():F3}");
                LogMessage($"[*] Step {stepCount}  CumReward: {GetCumulativeReward():F3}");

            }
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
            if (Input.GetKey(KeyCode.LeftShift)) thrust = 1f;

            contActions[0] = pitchx;
            contActions[1] = pitchz;
            contActions[2] = thrust;
        }

        private void OnTriggerEnter(Collider other)
        {
            // AstroScript'e değerse 
            if (other.TryGetComponent<AstroScript>(out AstroScript half))
            {
                AddReward(10f);
                counter += 10f;//eski hali =20, v1.2 güncellemesi
                astroRenderer.gameObject.GetComponent<SkinnedMeshRenderer>().enabled = false;
                astroCollider.gameObject.GetComponent<BoxCollider>().enabled = false;
                LogMessage("[Reward] Astro çarpması ödülü: 20");
                
                astroDestroyed = true;
                Debug.Log("Astro Collision, rewarded");
                EndEpisode();//v1.2 eklemesi
            }
            
            
        
            // Duvara çarparsa -1 ve bölüm sonu
            if (other.TryGetComponent<Wall>(out Wall fail))
            {
                //eski degerler reward : -1
                SetReward(-20f);
                LogMessage("[Reward] Duvar çarpması cezası: -20");
                EndEpisode();   
            }

            
        }
        
       
      
        // landzone'a belirlenen şartlarla inerse +0.5 ve bölüm sonu
         void OnCollisionEnter(Collision col)//triggerda bozuk calistigi icin OnCollisiona gecis yapildi
    {
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

    }

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
