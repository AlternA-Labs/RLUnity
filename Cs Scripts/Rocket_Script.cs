using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Serialization;

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
        [FormerlySerializedAs("AstroDestroyed")] [SerializeField] private bool astroDestroyed;
    
    
        [Header("Movement Settings")]
        [SerializeField] private float pitchSpeed = 100f;
        
        [SerializeField] private float thrustForce;//thrust force u delta time ile carpabilmek icin episode begine aldim.

        [Header("Penalties / Rewards")]
        [SerializeField] private float movePenalty = 0.001f;  // pitch kullanım cezası
        [SerializeField] private float stepPenalty = 0.0001f;  // Zaman cezası (her adım)
        [SerializeField] private float tiltPenalty = 0.001f;   // Yan yatma cezası (her adım)
        [FormerlySerializedAs("AlaboraPenalty")] [SerializeField] private float alaboraPenalty = 1f;    // Ters dönünce (tam alabora) verilecek ceza

        [Header("Stability Reward Settings")]
        [SerializeField] private float stableVelocityThreshold = 0.1f;
        [SerializeField] private float stableAngleThreshold = 5f;  // Kaç derecenin altı dik sayılacak
        [SerializeField] private float stableReward = 0.001f; 

        [Header("Approach Reward")]
        [SerializeField] private float approachRewardFactor = 0.01f;  //ekponansiyel ödül katsayısı
        [SerializeField] public float tiltThreshold = 10f;            // Başlangıç ceza eşiği
        [SerializeField] private float recoveryReward = 2f;        // Düzeltme ödül katsayısı
        [SerializeField] private float extremeTiltAngle = 80f;        // Aşırı sapma eşiği (örneğin 80 derece)
        [SerializeField] private float penaltyInterval = 1f;  // belirli bir açı için izin verilen süre
    
        // Önceki mesafeyi tutarak yaklaşma/uzaklaşma hesabı
        private float _previousDistanceToAstro = 0f;
        private float _tiltTimeAccumulator = 0f; 
        private float _nextPenaltyThreshold = 1f;

        // ReSharper disable Unity.PerformanceAnalysis
        public override void OnEpisodeBegin()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 70;
            // Bölüm (episode) başlangıcı
            thrustForce = 1000f * Time.deltaTime;
            transform.localPosition = new Vector3(0, 1f, 0);
            astroDestroyed = false;
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
                _previousDistanceToAstro = Vector3.Distance(transform.localPosition, astro.localPosition);
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
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Temel gözlemler
            sensor.AddObservation(transform.localPosition);
            sensor.AddObservation(transform.localRotation);
            sensor.AddObservation(rb.linearVelocity);

            // Astro konumunu da ekleyelim
            if (!astroDestroyed)
            {
                sensor.AddObservation(astro.localPosition);
            }
            else
            {
                sensor.AddObservation(landingSite.localPosition);//---> Buraya landing padin transfromunu koy yeni obje yaratip.
            }
        }

        // ReSharper disable Unity.PerformanceAnalysis
        public override void OnActionReceived(ActionBuffers actions)
        {
            float pitchInputX = actions.ContinuousActions[0];
            float pitchInputZ = actions.ContinuousActions[1];
            float thrustInput = actions.ContinuousActions[2];

            // Aksiyon sıfırdan farklıysa ufak ceza
            if (Mathf.Abs(pitchInputX) > 1e-6f) AddReward(-movePenalty);
            if (Mathf.Abs(pitchInputZ) > 1e-6f) AddReward(-movePenalty);
            // thrust için de ceza eklemek isterseniz yorumu açın:
            if (Mathf.Abs(thrustInput) > 1e-6f) AddReward(4 * movePenalty);
 
            // Roketi yönlendir
            transform.Rotate(pitchInputX * pitchSpeed * Time.deltaTime, 0f, 0f);
            transform.Rotate(0f, 0f, pitchInputZ * pitchSpeed * Time.deltaTime);
            rb.AddForce(thrustInput * thrustForce * transform.up);

            // Yan yatma
            float angleFromUp = Vector3.Angle(transform.up, Vector3.up);
            //Debug.Log($"angleFromUp: {angleFromUp}");
        


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
        
                    // Eğer açı extreme değerin üzerinde ise (bağımsız olarak ek ceza):
                    if (angleFromUp >= extremeTiltAngle)
                    {
                        AddReward(-alaboraPenalty);
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
                    AddReward(recoveryReward * recoveryTime);
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
            float minHeightThreshold = 1.2f;
            if (transform.localPosition.y < minHeightThreshold)
            {
                AddReward(-0.5f); // Hafif ceza, her adımda
                Debug.Log($"YErde durma cezası");
            }
            
            if (Vector3.Distance(transform.position, astro.position) < 1f && !astroDestroyed)//manuel carpisma
            {
                    OnTriggerEnter(astroCollider.GetComponent<Collider>());
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
                AddReward(1000f);
                astroRenderer.gameObject.GetComponent<SkinnedMeshRenderer>().enabled = false;
                astroCollider.gameObject.GetComponent<BoxCollider>().enabled = false;
                
                astroDestroyed = true;
                Debug.Log("Astro Collision, rewarded");
            }
            
            
        
            // Duvara çarparsa -1 ve bölüm sonu
            if (other.TryGetComponent<Wall>(out Wall fail))
            {
                //eski degerler reward : -1
                SetReward(-0.5f);
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
                AddReward(1000f);
                Debug.Log("Success");
                EndEpisode();    
            }
        }

    }

    void Update()
    {

        //roketlere uçmama cezası.

        /*
        if(transform.gameObject.CompareTag("land"))
        {
            AddReward(-1f); // İstediğin ceza miktarını ayarla.
            Debug.Log("Plane teması cezası verildi.");
        }
        */
        if (!astroDestroyed)
        {
            // Astro'ya yaklaşma ödülü (distance shaping)
            float currentDistance = Vector3.Distance(transform.position, astro.position);
            float distanceDelta = _previousDistanceToAstro - currentDistance;

            // Geçici değişken ile eski mesafeyi saklayalım
            float previousDistanceForLog = _previousDistanceToAstro;

            // Ödül hesaplaması
            //AddReward(distanceDelta * 5 * approachRewardFactor);

            // Roket eylemlerini uyguladıktan sonra:
            float speed = rb.linearVelocity.magnitude;
            float angleFromUp = Vector3.Angle(transform.up, Vector3.up);

            bool isApproaching = distanceDelta > 0;
            float sign = isApproaching ? 1f : -1f;
            float absDelta = Mathf.Abs(distanceDelta);

            float scalingFactor = 0.5f;
            float baseReward = Mathf.Exp(absDelta * scalingFactor) - 1f;
            float expReward = baseReward * approachRewardFactor * sign;

            AddReward(expReward);
            //Debug.Log($"Approach: {isApproaching} DistanceDelta: {distanceDelta}, ExpReward: {expReward} Previous: {previousDistanceForLog}, Current: {currentDistance}");
            

            // Eski mesafeyi güncelle
            _previousDistanceToAstro = currentDistance;
        }
    }

    }
}
