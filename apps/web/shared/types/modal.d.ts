import type { AllowedComponentProps, VNodeProps } from "vue";

declare global {
    namespace Modal {
        type ComponentLoader<C extends Component> = () => Promise<{
            default: C;
        }>;

        type ComponentProps<C extends Component> = C extends new (...args: any) => any
            ? Omit<InstanceType<C>["$props"], keyof VNodeProps | keyof AllowedComponentProps>
            : never;

        interface Args<C extends Component> {
            loader: ComponentLoader<C>;
            props?: ComponentProps<C>;
            key: string;
        }
    }
}
