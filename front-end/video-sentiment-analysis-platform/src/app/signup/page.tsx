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
        <div className="min-h-screen bg-white">
      <nav className="flex h-16 items-center justify-between border-b border-gray-200 px-10">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-gray-800 text-white">
            SA
          </div>
          <span className="text-lg font-medium">Sentiment Analysis</span>
        </div>
      </nav>

      <main className="flex h-[calc(100vh-4rem)] items-center justify-center">
        <div className="w-full max-w-md space-y-8 px-4">
          <div className="text-center">
            <h2 className="text-2xl font-bold">Create an account</h2>
            <p className="mt-2 text-sm text-gray-600">
              Sign up to get started with Sentiment Analysis
            </p>
          </div>
        </div>

            <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="mt-8 w-full max-w-md space-y-6 px-4"
            >
                <div className="space-y-4">
                    <div>
                        <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                            Name
                        </label>
                        <input
                            type="text"
                            id="name"
                            {...form.register("name")}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        />
                        {form.formState.errors.name && (
                            <p className="mt-1 text-sm text-red-600">
                                {form.formState.errors.name.message}
                            </p>
                        )}
                    </div>

                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                            Email address
                        </label>
                        <input
                            type="email"
                            id="email"
                            {...form.register("email")}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        />
                        {form.formState.errors.email && (
                            <p className="mt-1 text-sm text-red-600">
                                {form.formState.errors.email.message}
                            </p>
                        )}
                    </div>

                    <div>
                        <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                            Password
                        </label>
                        <input
                            type="password"
                            id="password"
                            {...form.register("password")}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        />
                        {form.formState.errors.password && (
                            <p className="mt-1 text-sm text-red-600">
                                {form.formState.errors.password.message}
                            </p>
                        )}
                    </div>

                    <div>
                        <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                            Confirm Password
                        </label>
                        <input
                            type="password"
                            id="confirmPassword"
                            {...form.register("confirmPassword")}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        />
                        {form.formState.errors.confirmPassword && (
                            <p className="mt-1 text-sm text-red-600">
                                {form.formState.errors.confirmPassword.message}
                            </p>
                        )}
                    </div>
                </div>

                <div>
                    <button
                        type="submit"
                        className="flex w-full justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                    >
                        Sign Up
                    </button>
                </div>
            </form>
        </main>
        </div>
    )
}