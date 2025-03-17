using UnityEngine;

namespace RLUnity.Cs_Scripts
{
    public class AstroScript : MonoBehaviour
    {
        void Update()
        {
            transform.Rotate(0f, 100f * Time.deltaTime, 0f);
        }
    }
}
