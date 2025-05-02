using System;
using UnityEngine;
using Debug = UnityEngine.Debug;
namespace RLUnity.Cs_Scripts
{
    
    public class AstroSensorCollision : MonoBehaviour
    {
        public Transform rocketRoot;   // Inspector’dan RocketAgent transformunu atayın
        public Vector3 localOffset = new Vector3(0f, 0.8f, 0f);

        void Update()          // her karede roketten sonra çalışır
        {
            transform.position = rocketRoot.TransformPoint(localOffset);
            transform.rotation = rocketRoot.rotation;   // istersen bakış yönü de eşlensin
        }
        
        public RocketAgent agent;   // Inspector’dan sürükle-bırak
        void OnCollisionEnter(Collision col)
        {
            // Yüzeye “Astro” tag’i verdiysen doğrudan:
            if (col.gameObject.CompareTag("Astro"))
            {
                Debug.Log("Astro’yu çarptık!");
                agent.OnAstroHit();
            }
        }

    }
}