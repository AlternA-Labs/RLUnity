using UnityEngine;

public class AstroScript : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        transform.Rotate(0f, 100f * Time.deltaTime, 0f);
    }

    private void OnTriggerEnter(Collider other)
    {
        Destroy(gameObject);
    }

}
