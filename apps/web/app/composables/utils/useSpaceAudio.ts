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

            // Add listeners for various user interaction events
            const startAudio = async () => {
                if (audioElement.value && !isPlaying.value) {
                    try {
                        await audioElement.value.play();
                        isPlaying.value = true;
                        removeListeners();
                    } catch (e) {
                        console.error("Failed to play audio:", e);
                    }
                }
            };

            const removeListeners = () => {
                document.removeEventListener("click", startAudio, true);
                document.removeEventListener("keydown", startAudio, true);
                document.removeEventListener("touchstart", startAudio, true);
                document.removeEventListener("mousedown", startAudio, true);
                document.removeEventListener("scroll", startAudio, true);
            };

            // Use capture phase for more reliable event catching
            document.addEventListener("click", startAudio, { capture: true, once: true });
            document.addEventListener("keydown", startAudio, { capture: true, once: true });
            document.addEventListener("touchstart", startAudio, { capture: true, once: true });
            document.addEventListener("mousedown", startAudio, { capture: true, once: true });
            document.addEventListener("scroll", startAudio, { capture: true, once: true });
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
