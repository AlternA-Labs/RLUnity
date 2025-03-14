using UnityEngine;

public class timerScript : MonoBehaviour
{
    [Range(1f, 100f)] 
    public float timeScale = 10f; //* Time.deltaTime;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Time.timeScale = timeScale;

        // Ayrıca frame hızını sınırsız yapmak için:
        Application.targetFrameRate = -1;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
