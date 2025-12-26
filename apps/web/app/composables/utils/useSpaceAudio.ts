/**
 * Composable for managing space ambient audio
 * Uses singleton pattern to ensure only one audio instance exists
 */

// Singleton state - shared across all component instances
const audioElement = shallowRef<HTMLAudioElement | null>(null);
const isMuted = ref(false);
const isPlaying = ref(false);
const volume = ref(0.3); // Default volume 30%
const isInitialized = ref(false);

export default function () {
    function init() {
        if (typeof window === "undefined") return;
        if (isInitialized.value) return; // Already initialized

        isInitialized.value = true;

        // Create audio element
        const audio = new Audio("/audio/nasaSound.mp3");
        audio.loop = true;
        audio.volume = volume.value;
        audioElement.value = audio;

        // Try to autoplay (may be blocked by browser)
        tryAutoplay();
    }

    async function tryAutoplay() {
        if (!audioElement.value) return;

        try {
            await audioElement.value.play();
            isPlaying.value = true;
        } catch {
            // Autoplay blocked, will need user interaction
            console.log("Autoplay blocked, waiting for user interaction");

            // Add one-time click listener to start audio
            const startAudio = async () => {
                if (audioElement.value && !isPlaying.value) {
                    try {
                        await audioElement.value.play();
                        isPlaying.value = true;
                    } catch (e) {
                        console.error("Failed to play audio:", e);
                    }
                }
                document.removeEventListener("click", startAudio);
                document.removeEventListener("keydown", startAudio);
            };

            document.addEventListener("click", startAudio, { once: true });
            document.addEventListener("keydown", startAudio, { once: true });
        }
    }

    function toggleMute() {
        if (!audioElement.value) return;

        isMuted.value = !isMuted.value;
        audioElement.value.muted = isMuted.value;
    }

    function setVolume(newVolume: number) {
        if (!audioElement.value) return;

        volume.value = Math.max(0, Math.min(1, newVolume));
        audioElement.value.volume = volume.value;
    }

    function play() {
        if (!audioElement.value) return;

        audioElement.value.play();
        isPlaying.value = true;
    }

    function pause() {
        if (!audioElement.value) return;

        audioElement.value.pause();
        isPlaying.value = false;
    }

    function cleanup() {
        if (audioElement.value) {
            audioElement.value.pause();
            audioElement.value.src = "";
            audioElement.value = null;
        }
        isPlaying.value = false;
        isInitialized.value = false;
    }

    return {
        audioElement,
        isMuted,
        isPlaying,
        volume,
        init,
        toggleMute,
        setVolume,
        play,
        pause,
        cleanup,
    };
}
