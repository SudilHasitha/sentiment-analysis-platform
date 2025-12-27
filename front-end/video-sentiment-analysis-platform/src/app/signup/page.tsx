"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { signupSchema } from "~/schemas/auth";

export default function SignupPage(){

    const form = useForm<signupSchema>({
        resolver: zodResolver(signupSchema),
        defaultValues: {
            name: "",
            email: "",
            password: "",
            confirmPassword: "",
        },
    });

    const onSubmit = (data: signupSchema) => {
        console.log(data);
    }

    return (
        <div>
            <h1>Signup Page</h1>
        </div>
    )
}