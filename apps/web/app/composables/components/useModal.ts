import type { ShallowRef } from "vue";

export default function () {
    const modalComponents = useState<ShallowRef<Component | null>[]>("modal:components", () =>
        Array.from({ length: CONSTANTS.COMPOSABLES.USE_MODAL.STORAGE_LENGTH }));
    const currentComponentIndex = useState("modal:current-index", () => 0);
    const currentModalKey = useState<string | undefined>("modal:current-component:key");

    const open = useState("modal:open", () => false);
    const componentProps = useState<object>("modal:component-props");

    const currentComponent = computed<ShallowRef<Component | null> | null>(() => {
        if (!modalComponents.value)
            return null;

        const index = currentComponentIndex.value;
        if (index < 0 || index > modalComponents.value.length)
            return null;

        return modalComponents.value[index] ?? null;
    });

    function nextIndex() {
        if (!modalComponents.value)
            return;

        currentComponentIndex.value
            = (currentComponentIndex.value + 1) % CONSTANTS.COMPOSABLES.USE_MODAL.STORAGE_LENGTH;
    }

    function loadComponent<C extends Component>({ loader, props, key }: Modal.Args<C>) {
        const component = defineAsyncComponent({
            loader,
            delay: 0,
            timeout: 5000,
        });

        nextIndex();

        currentModalKey.value = key;

        nextTick(() => {
            modalComponents.value[currentComponentIndex.value] = shallowRef<Component>(
                markRaw(component),
            );

            componentProps.value = props as object;
        });
    }

    return {
        modalComponents,
        componentProps,
        currentModalKey: readonly(currentModalKey),
        currentComponent,
        currentComponentIndex,
        loadComponent,
        open,
    };
}
