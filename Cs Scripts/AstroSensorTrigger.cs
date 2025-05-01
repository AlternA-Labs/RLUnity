using UnityEngine;

namespace RLUnity.Cs_Scripts
{
    public class AstroSensorCollision : MonoBehaviour
    {
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